import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_backbone_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_inputs_: torch.Tensor,
        L_self_modules_backbone_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_conv_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_conv_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_conv_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_conv_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_conv_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_conv_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_conv_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_conv_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_conv_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_conv_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_decode_head_modules_aspp_modules_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_aspp_modules_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_aspp_modules_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_aspp_modules_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_aspp_modules_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_aspp_modules_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_aspp_modules_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_aspp_modules_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_aspp_modules_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_aspp_modules_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_aspp_modules_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_aspp_modules_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_aspp_modules_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_aspp_modules_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_aspp_modules_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_bottleneck_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_conv_seg_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_conv_seg_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_backbone_modules_conv1_modules_conv_parameters_weight_ = (
            L_self_modules_backbone_modules_conv1_modules_conv_parameters_weight_
        )
        l_inputs_ = L_inputs_
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
        l_self_modules_backbone_modules_layer1_modules_0_modules_conv_modules_0_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_conv_modules_0_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_conv_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_0_modules_conv_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_0_modules_conv_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_0_modules_conv_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_0_modules_conv_modules_0_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_conv_modules_0_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_conv_modules_0_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_0_modules_conv_modules_0_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_0_modules_conv_modules_1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_conv_modules_1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_conv_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_0_modules_conv_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_0_modules_conv_modules_1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_0_modules_conv_modules_1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_0_modules_conv_modules_1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_conv_modules_1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_conv_modules_1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_0_modules_conv_modules_1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_0_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_0_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_0_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_0_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_0_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_0_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_0_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_0_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_0_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_0_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_0_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_0_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_0_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_0_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_0_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_0_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_0_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_0_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_0_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_0_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_0_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_0_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_0_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_0_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_0_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_0_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_0_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_0_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_0_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_0_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_0_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_0_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_0_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_0_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_0_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_0_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_0_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_0_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_0_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_0_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_0_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_0_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_0_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_0_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_0_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_0_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_0_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_0_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_0_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_0_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_0_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_0_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_0_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_0_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_0_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_0_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_0_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_0_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_0_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_0_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_0_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_0_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_0_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_0_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_0_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_0_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_0_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_0_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_0_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_0_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_0_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_0_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_0_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_0_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_0_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_0_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_0_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_0_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_0_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_0_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_0_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_0_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_0_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_0_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_0_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_0_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_0_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_0_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_0_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_0_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_0_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_0_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_0_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_0_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_0_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_0_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_2_modules_bn_parameters_bias_
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
        l_self_modules_decode_head_modules_aspp_modules_modules_1_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_aspp_modules_modules_1_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_aspp_modules_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_aspp_modules_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_aspp_modules_modules_1_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_aspp_modules_modules_1_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_aspp_modules_modules_1_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_aspp_modules_modules_1_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_aspp_modules_modules_1_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_aspp_modules_modules_1_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_aspp_modules_modules_2_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_aspp_modules_modules_2_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_aspp_modules_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_aspp_modules_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_aspp_modules_modules_2_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_aspp_modules_modules_2_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_aspp_modules_modules_2_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_aspp_modules_modules_2_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_aspp_modules_modules_2_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_aspp_modules_modules_2_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_aspp_modules_modules_3_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_aspp_modules_modules_3_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_aspp_modules_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_aspp_modules_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_aspp_modules_modules_3_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_aspp_modules_modules_3_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_aspp_modules_modules_3_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_aspp_modules_modules_3_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_aspp_modules_modules_3_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_aspp_modules_modules_3_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_bottleneck_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_bottleneck_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_weight_ = (
            L_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_weight_
        )
        l_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_bias_ = (
            L_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_bias_
        )
        l_self_modules_decode_head_modules_conv_seg_parameters_weight_ = (
            L_self_modules_decode_head_modules_conv_seg_parameters_weight_
        )
        l_self_modules_decode_head_modules_conv_seg_parameters_bias_ = (
            L_self_modules_decode_head_modules_conv_seg_parameters_bias_
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
        x_2 = torch.nn.functional.hardtanh(x_1, 0.0, 6.0, True)
        x_1 = None
        x_3 = torch.conv2d(
            x_2,
            l_self_modules_backbone_modules_layer1_modules_0_modules_conv_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_2 = l_self_modules_backbone_modules_layer1_modules_0_modules_conv_modules_0_modules_conv_parameters_weight_ = (None)
        x_4 = torch.nn.functional.batch_norm(
            x_3,
            l_self_modules_backbone_modules_layer1_modules_0_modules_conv_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_conv_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_conv_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_conv_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_3 = l_self_modules_backbone_modules_layer1_modules_0_modules_conv_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_0_modules_conv_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_0_modules_conv_modules_0_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_0_modules_conv_modules_0_modules_bn_parameters_bias_ = (None)
        x_5 = torch.nn.functional.hardtanh(x_4, 0.0, 6.0, True)
        x_4 = None
        x_6 = torch.conv2d(
            x_5,
            l_self_modules_backbone_modules_layer1_modules_0_modules_conv_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_5 = l_self_modules_backbone_modules_layer1_modules_0_modules_conv_modules_1_modules_conv_parameters_weight_ = (None)
        x_7 = torch.nn.functional.batch_norm(
            x_6,
            l_self_modules_backbone_modules_layer1_modules_0_modules_conv_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_conv_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_conv_modules_1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_conv_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_6 = l_self_modules_backbone_modules_layer1_modules_0_modules_conv_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_0_modules_conv_modules_1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_0_modules_conv_modules_1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_0_modules_conv_modules_1_modules_bn_parameters_bias_ = (None)
        x_8 = torch.conv2d(
            x_7,
            l_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_7 = l_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_0_modules_conv_parameters_weight_ = (None)
        x_9 = torch.nn.functional.batch_norm(
            x_8,
            l_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_8 = l_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_0_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_0_modules_bn_parameters_bias_ = (None)
        x_10 = torch.nn.functional.hardtanh(x_9, 0.0, 6.0, True)
        x_9 = None
        x_11 = torch.conv2d(
            x_10,
            l_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            96,
        )
        x_10 = l_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_1_modules_conv_parameters_weight_ = (None)
        x_12 = torch.nn.functional.batch_norm(
            x_11,
            l_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_11 = l_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_1_modules_bn_parameters_bias_ = (None)
        x_13 = torch.nn.functional.hardtanh(x_12, 0.0, 6.0, True)
        x_12 = None
        x_14 = torch.conv2d(
            x_13,
            l_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_13 = l_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_2_modules_conv_parameters_weight_ = (None)
        x_15 = torch.nn.functional.batch_norm(
            x_14,
            l_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_14 = l_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_0_modules_conv_modules_2_modules_bn_parameters_bias_ = (None)
        x_16 = torch.conv2d(
            x_15,
            l_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_17 = torch.nn.functional.batch_norm(
            x_16,
            l_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_16 = l_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_0_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_0_modules_bn_parameters_bias_ = (None)
        x_18 = torch.nn.functional.hardtanh(x_17, 0.0, 6.0, True)
        x_17 = None
        x_19 = torch.conv2d(
            x_18,
            l_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            144,
        )
        x_18 = l_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_1_modules_conv_parameters_weight_ = (None)
        x_20 = torch.nn.functional.batch_norm(
            x_19,
            l_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_19 = l_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_1_modules_bn_parameters_bias_ = (None)
        x_21 = torch.nn.functional.hardtanh(x_20, 0.0, 6.0, True)
        x_20 = None
        x_22 = torch.conv2d(
            x_21,
            l_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_21 = l_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_2_modules_conv_parameters_weight_ = (None)
        x_23 = torch.nn.functional.batch_norm(
            x_22,
            l_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_22 = l_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_1_modules_conv_modules_2_modules_bn_parameters_bias_ = (None)
        out = x_15 + x_23
        x_15 = x_23 = None
        x_24 = torch.conv2d(
            out,
            l_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out = l_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_0_modules_conv_parameters_weight_ = (None)
        x_25 = torch.nn.functional.batch_norm(
            x_24,
            l_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_24 = l_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_0_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_0_modules_bn_parameters_bias_ = (None)
        x_26 = torch.nn.functional.hardtanh(x_25, 0.0, 6.0, True)
        x_25 = None
        x_27 = torch.conv2d(
            x_26,
            l_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            144,
        )
        x_26 = l_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_1_modules_conv_parameters_weight_ = (None)
        x_28 = torch.nn.functional.batch_norm(
            x_27,
            l_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_27 = l_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_1_modules_bn_parameters_bias_ = (None)
        x_29 = torch.nn.functional.hardtanh(x_28, 0.0, 6.0, True)
        x_28 = None
        x_30 = torch.conv2d(
            x_29,
            l_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_29 = l_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_2_modules_conv_parameters_weight_ = (None)
        x_31 = torch.nn.functional.batch_norm(
            x_30,
            l_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_30 = l_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_conv_modules_2_modules_bn_parameters_bias_ = (None)
        x_32 = torch.conv2d(
            x_31,
            l_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_33 = torch.nn.functional.batch_norm(
            x_32,
            l_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_32 = l_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_0_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_0_modules_bn_parameters_bias_ = (None)
        x_34 = torch.nn.functional.hardtanh(x_33, 0.0, 6.0, True)
        x_33 = None
        x_35 = torch.conv2d(
            x_34,
            l_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            192,
        )
        x_34 = l_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_1_modules_conv_parameters_weight_ = (None)
        x_36 = torch.nn.functional.batch_norm(
            x_35,
            l_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_35 = l_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_1_modules_bn_parameters_bias_ = (None)
        x_37 = torch.nn.functional.hardtanh(x_36, 0.0, 6.0, True)
        x_36 = None
        x_38 = torch.conv2d(
            x_37,
            l_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_37 = l_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_2_modules_conv_parameters_weight_ = (None)
        x_39 = torch.nn.functional.batch_norm(
            x_38,
            l_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_38 = l_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_1_modules_conv_modules_2_modules_bn_parameters_bias_ = (None)
        out_1 = x_31 + x_39
        x_31 = x_39 = None
        x_40 = torch.conv2d(
            out_1,
            l_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_41 = torch.nn.functional.batch_norm(
            x_40,
            l_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_40 = l_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_0_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_0_modules_bn_parameters_bias_ = (None)
        x_42 = torch.nn.functional.hardtanh(x_41, 0.0, 6.0, True)
        x_41 = None
        x_43 = torch.conv2d(
            x_42,
            l_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            192,
        )
        x_42 = l_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_1_modules_conv_parameters_weight_ = (None)
        x_44 = torch.nn.functional.batch_norm(
            x_43,
            l_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_43 = l_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_1_modules_bn_parameters_bias_ = (None)
        x_45 = torch.nn.functional.hardtanh(x_44, 0.0, 6.0, True)
        x_44 = None
        x_46 = torch.conv2d(
            x_45,
            l_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_45 = l_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_2_modules_conv_parameters_weight_ = (None)
        x_47 = torch.nn.functional.batch_norm(
            x_46,
            l_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_46 = l_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_2_modules_conv_modules_2_modules_bn_parameters_bias_ = (None)
        out_2 = out_1 + x_47
        out_1 = x_47 = None
        x_48 = torch.conv2d(
            out_2,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_2 = l_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_0_modules_conv_parameters_weight_ = (None)
        x_49 = torch.nn.functional.batch_norm(
            x_48,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_48 = l_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_0_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_0_modules_bn_parameters_bias_ = (None)
        x_50 = torch.nn.functional.hardtanh(x_49, 0.0, 6.0, True)
        x_49 = None
        x_51 = torch.conv2d(
            x_50,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (2, 2),
            192,
        )
        x_50 = l_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_1_modules_conv_parameters_weight_ = (None)
        x_52 = torch.nn.functional.batch_norm(
            x_51,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_51 = l_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_1_modules_bn_parameters_bias_ = (None)
        x_53 = torch.nn.functional.hardtanh(x_52, 0.0, 6.0, True)
        x_52 = None
        x_54 = torch.conv2d(
            x_53,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_53 = l_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_2_modules_conv_parameters_weight_ = (None)
        x_55 = torch.nn.functional.batch_norm(
            x_54,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_54 = l_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_conv_modules_2_modules_bn_parameters_bias_ = (None)
        x_56 = torch.conv2d(
            x_55,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_57 = torch.nn.functional.batch_norm(
            x_56,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_56 = l_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_0_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_0_modules_bn_parameters_bias_ = (None)
        x_58 = torch.nn.functional.hardtanh(x_57, 0.0, 6.0, True)
        x_57 = None
        x_59 = torch.conv2d(
            x_58,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        x_58 = l_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_1_modules_conv_parameters_weight_ = (None)
        x_60 = torch.nn.functional.batch_norm(
            x_59,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_59 = l_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_1_modules_bn_parameters_bias_ = (None)
        x_61 = torch.nn.functional.hardtanh(x_60, 0.0, 6.0, True)
        x_60 = None
        x_62 = torch.conv2d(
            x_61,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_61 = l_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_2_modules_conv_parameters_weight_ = (None)
        x_63 = torch.nn.functional.batch_norm(
            x_62,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_62 = l_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_1_modules_conv_modules_2_modules_bn_parameters_bias_ = (None)
        out_3 = x_55 + x_63
        x_55 = x_63 = None
        x_64 = torch.conv2d(
            out_3,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_65 = torch.nn.functional.batch_norm(
            x_64,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_64 = l_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_0_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_0_modules_bn_parameters_bias_ = (None)
        x_66 = torch.nn.functional.hardtanh(x_65, 0.0, 6.0, True)
        x_65 = None
        x_67 = torch.conv2d(
            x_66,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        x_66 = l_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_1_modules_conv_parameters_weight_ = (None)
        x_68 = torch.nn.functional.batch_norm(
            x_67,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_67 = l_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_1_modules_bn_parameters_bias_ = (None)
        x_69 = torch.nn.functional.hardtanh(x_68, 0.0, 6.0, True)
        x_68 = None
        x_70 = torch.conv2d(
            x_69,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_69 = l_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_2_modules_conv_parameters_weight_ = (None)
        x_71 = torch.nn.functional.batch_norm(
            x_70,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_70 = l_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_2_modules_conv_modules_2_modules_bn_parameters_bias_ = (None)
        out_4 = out_3 + x_71
        out_3 = x_71 = None
        x_72 = torch.conv2d(
            out_4,
            l_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_73 = torch.nn.functional.batch_norm(
            x_72,
            l_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_72 = l_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_0_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_0_modules_bn_parameters_bias_ = (None)
        x_74 = torch.nn.functional.hardtanh(x_73, 0.0, 6.0, True)
        x_73 = None
        x_75 = torch.conv2d(
            x_74,
            l_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        x_74 = l_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_1_modules_conv_parameters_weight_ = (None)
        x_76 = torch.nn.functional.batch_norm(
            x_75,
            l_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_75 = l_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_1_modules_bn_parameters_bias_ = (None)
        x_77 = torch.nn.functional.hardtanh(x_76, 0.0, 6.0, True)
        x_76 = None
        x_78 = torch.conv2d(
            x_77,
            l_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_77 = l_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_2_modules_conv_parameters_weight_ = (None)
        x_79 = torch.nn.functional.batch_norm(
            x_78,
            l_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_78 = l_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_3_modules_conv_modules_2_modules_bn_parameters_bias_ = (None)
        out_5 = out_4 + x_79
        out_4 = x_79 = None
        x_80 = torch.conv2d(
            out_5,
            l_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_5 = l_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_0_modules_conv_parameters_weight_ = (None)
        x_81 = torch.nn.functional.batch_norm(
            x_80,
            l_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_80 = l_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_0_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_0_modules_bn_parameters_bias_ = (None)
        x_82 = torch.nn.functional.hardtanh(x_81, 0.0, 6.0, True)
        x_81 = None
        x_83 = torch.conv2d(
            x_82,
            l_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (2, 2),
            384,
        )
        x_82 = l_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_1_modules_conv_parameters_weight_ = (None)
        x_84 = torch.nn.functional.batch_norm(
            x_83,
            l_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_83 = l_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_1_modules_bn_parameters_bias_ = (None)
        x_85 = torch.nn.functional.hardtanh(x_84, 0.0, 6.0, True)
        x_84 = None
        x_86 = torch.conv2d(
            x_85,
            l_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_85 = l_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_2_modules_conv_parameters_weight_ = (None)
        x_87 = torch.nn.functional.batch_norm(
            x_86,
            l_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_86 = l_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer5_modules_0_modules_conv_modules_2_modules_bn_parameters_bias_ = (None)
        x_88 = torch.conv2d(
            x_87,
            l_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_89 = torch.nn.functional.batch_norm(
            x_88,
            l_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_88 = l_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_0_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_0_modules_bn_parameters_bias_ = (None)
        x_90 = torch.nn.functional.hardtanh(x_89, 0.0, 6.0, True)
        x_89 = None
        x_91 = torch.conv2d(
            x_90,
            l_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            576,
        )
        x_90 = l_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_1_modules_conv_parameters_weight_ = (None)
        x_92 = torch.nn.functional.batch_norm(
            x_91,
            l_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_91 = l_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_1_modules_bn_parameters_bias_ = (None)
        x_93 = torch.nn.functional.hardtanh(x_92, 0.0, 6.0, True)
        x_92 = None
        x_94 = torch.conv2d(
            x_93,
            l_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_93 = l_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_2_modules_conv_parameters_weight_ = (None)
        x_95 = torch.nn.functional.batch_norm(
            x_94,
            l_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_94 = l_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer5_modules_1_modules_conv_modules_2_modules_bn_parameters_bias_ = (None)
        out_6 = x_87 + x_95
        x_87 = x_95 = None
        x_96 = torch.conv2d(
            out_6,
            l_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_97 = torch.nn.functional.batch_norm(
            x_96,
            l_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_96 = l_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_0_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_0_modules_bn_parameters_bias_ = (None)
        x_98 = torch.nn.functional.hardtanh(x_97, 0.0, 6.0, True)
        x_97 = None
        x_99 = torch.conv2d(
            x_98,
            l_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            576,
        )
        x_98 = l_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_1_modules_conv_parameters_weight_ = (None)
        x_100 = torch.nn.functional.batch_norm(
            x_99,
            l_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_99 = l_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_1_modules_bn_parameters_bias_ = (None)
        x_101 = torch.nn.functional.hardtanh(x_100, 0.0, 6.0, True)
        x_100 = None
        x_102 = torch.conv2d(
            x_101,
            l_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_101 = l_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_2_modules_conv_parameters_weight_ = (None)
        x_103 = torch.nn.functional.batch_norm(
            x_102,
            l_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_102 = l_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer5_modules_2_modules_conv_modules_2_modules_bn_parameters_bias_ = (None)
        out_7 = out_6 + x_103
        out_6 = x_103 = None
        x_104 = torch.conv2d(
            out_7,
            l_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_7 = l_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_0_modules_conv_parameters_weight_ = (None)
        x_105 = torch.nn.functional.batch_norm(
            x_104,
            l_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_104 = l_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_0_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_0_modules_bn_parameters_bias_ = (None)
        x_106 = torch.nn.functional.hardtanh(x_105, 0.0, 6.0, True)
        x_105 = None
        x_107 = torch.conv2d(
            x_106,
            l_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (4, 4),
            576,
        )
        x_106 = l_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_1_modules_conv_parameters_weight_ = (None)
        x_108 = torch.nn.functional.batch_norm(
            x_107,
            l_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_107 = l_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_1_modules_bn_parameters_bias_ = (None)
        x_109 = torch.nn.functional.hardtanh(x_108, 0.0, 6.0, True)
        x_108 = None
        x_110 = torch.conv2d(
            x_109,
            l_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_109 = l_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_2_modules_conv_parameters_weight_ = (None)
        x_111 = torch.nn.functional.batch_norm(
            x_110,
            l_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_110 = l_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer6_modules_0_modules_conv_modules_2_modules_bn_parameters_bias_ = (None)
        x_112 = torch.conv2d(
            x_111,
            l_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_113 = torch.nn.functional.batch_norm(
            x_112,
            l_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_112 = l_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_0_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_0_modules_bn_parameters_bias_ = (None)
        x_114 = torch.nn.functional.hardtanh(x_113, 0.0, 6.0, True)
        x_113 = None
        x_115 = torch.conv2d(
            x_114,
            l_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            960,
        )
        x_114 = l_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_1_modules_conv_parameters_weight_ = (None)
        x_116 = torch.nn.functional.batch_norm(
            x_115,
            l_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_115 = l_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_1_modules_bn_parameters_bias_ = (None)
        x_117 = torch.nn.functional.hardtanh(x_116, 0.0, 6.0, True)
        x_116 = None
        x_118 = torch.conv2d(
            x_117,
            l_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_117 = l_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_2_modules_conv_parameters_weight_ = (None)
        x_119 = torch.nn.functional.batch_norm(
            x_118,
            l_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_118 = l_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer6_modules_1_modules_conv_modules_2_modules_bn_parameters_bias_ = (None)
        out_8 = x_111 + x_119
        x_111 = x_119 = None
        x_120 = torch.conv2d(
            out_8,
            l_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_121 = torch.nn.functional.batch_norm(
            x_120,
            l_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_120 = l_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_0_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_0_modules_bn_parameters_bias_ = (None)
        x_122 = torch.nn.functional.hardtanh(x_121, 0.0, 6.0, True)
        x_121 = None
        x_123 = torch.conv2d(
            x_122,
            l_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            960,
        )
        x_122 = l_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_1_modules_conv_parameters_weight_ = (None)
        x_124 = torch.nn.functional.batch_norm(
            x_123,
            l_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_123 = l_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_1_modules_bn_parameters_bias_ = (None)
        x_125 = torch.nn.functional.hardtanh(x_124, 0.0, 6.0, True)
        x_124 = None
        x_126 = torch.conv2d(
            x_125,
            l_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_125 = l_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_2_modules_conv_parameters_weight_ = (None)
        x_127 = torch.nn.functional.batch_norm(
            x_126,
            l_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_126 = l_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer6_modules_2_modules_conv_modules_2_modules_bn_parameters_bias_ = (None)
        out_9 = out_8 + x_127
        out_8 = x_127 = None
        x_128 = torch.conv2d(
            out_9,
            l_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_9 = l_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_0_modules_conv_parameters_weight_ = (None)
        x_129 = torch.nn.functional.batch_norm(
            x_128,
            l_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_128 = l_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_0_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_0_modules_bn_parameters_bias_ = (None)
        x_130 = torch.nn.functional.hardtanh(x_129, 0.0, 6.0, True)
        x_129 = None
        x_131 = torch.conv2d(
            x_130,
            l_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (4, 4),
            960,
        )
        x_130 = l_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_1_modules_conv_parameters_weight_ = (None)
        x_132 = torch.nn.functional.batch_norm(
            x_131,
            l_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_131 = l_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_1_modules_bn_parameters_bias_ = (None)
        x_133 = torch.nn.functional.hardtanh(x_132, 0.0, 6.0, True)
        x_132 = None
        x_134 = torch.conv2d(
            x_133,
            l_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_133 = l_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_2_modules_conv_parameters_weight_ = (None)
        x_135 = torch.nn.functional.batch_norm(
            x_134,
            l_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_134 = l_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_layer7_modules_0_modules_conv_modules_2_modules_bn_parameters_bias_ = (None)
        input_1 = torch.nn.functional.adaptive_avg_pool2d(x_135, 1)
        x_136 = torch.conv2d(
            input_1,
            l_self_modules_decode_head_modules_image_pool_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_1 = l_self_modules_decode_head_modules_image_pool_modules_1_modules_conv_parameters_weight_ = (None)
        x_137 = torch.nn.functional.batch_norm(
            x_136,
            l_self_modules_decode_head_modules_image_pool_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_image_pool_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_image_pool_modules_1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_image_pool_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_136 = l_self_modules_decode_head_modules_image_pool_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_image_pool_modules_1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_image_pool_modules_1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_image_pool_modules_1_modules_bn_parameters_bias_ = (None)
        x_138 = torch.nn.functional.relu(x_137, inplace=True)
        x_137 = None
        interpolate = torch.nn.functional.interpolate(
            x_138, (64, 64), None, "bilinear", False
        )
        x_138 = None
        x_139 = torch.conv2d(
            x_135,
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
        x_140 = torch.nn.functional.batch_norm(
            x_139,
            l_self_modules_decode_head_modules_aspp_modules_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_aspp_modules_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_aspp_modules_modules_0_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_aspp_modules_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_139 = l_self_modules_decode_head_modules_aspp_modules_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_aspp_modules_modules_0_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_aspp_modules_modules_0_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_aspp_modules_modules_0_modules_bn_parameters_bias_ = (None)
        x_141 = torch.nn.functional.relu(x_140, inplace=True)
        x_140 = None
        x_142 = torch.conv2d(
            x_135,
            l_self_modules_decode_head_modules_aspp_modules_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (12, 12),
            (12, 12),
            1,
        )
        l_self_modules_decode_head_modules_aspp_modules_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_143 = torch.nn.functional.batch_norm(
            x_142,
            l_self_modules_decode_head_modules_aspp_modules_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_aspp_modules_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_aspp_modules_modules_1_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_aspp_modules_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_142 = l_self_modules_decode_head_modules_aspp_modules_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_aspp_modules_modules_1_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_aspp_modules_modules_1_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_aspp_modules_modules_1_modules_bn_parameters_bias_ = (None)
        x_144 = torch.nn.functional.relu(x_143, inplace=True)
        x_143 = None
        x_145 = torch.conv2d(
            x_135,
            l_self_modules_decode_head_modules_aspp_modules_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (24, 24),
            (24, 24),
            1,
        )
        l_self_modules_decode_head_modules_aspp_modules_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_146 = torch.nn.functional.batch_norm(
            x_145,
            l_self_modules_decode_head_modules_aspp_modules_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_aspp_modules_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_aspp_modules_modules_2_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_aspp_modules_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_145 = l_self_modules_decode_head_modules_aspp_modules_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_aspp_modules_modules_2_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_aspp_modules_modules_2_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_aspp_modules_modules_2_modules_bn_parameters_bias_ = (None)
        x_147 = torch.nn.functional.relu(x_146, inplace=True)
        x_146 = None
        x_148 = torch.conv2d(
            x_135,
            l_self_modules_decode_head_modules_aspp_modules_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (36, 36),
            (36, 36),
            1,
        )
        x_135 = l_self_modules_decode_head_modules_aspp_modules_modules_3_modules_conv_parameters_weight_ = (None)
        x_149 = torch.nn.functional.batch_norm(
            x_148,
            l_self_modules_decode_head_modules_aspp_modules_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_aspp_modules_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_aspp_modules_modules_3_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_aspp_modules_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_148 = l_self_modules_decode_head_modules_aspp_modules_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_aspp_modules_modules_3_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_aspp_modules_modules_3_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_aspp_modules_modules_3_modules_bn_parameters_bias_ = (None)
        x_150 = torch.nn.functional.relu(x_149, inplace=True)
        x_149 = None
        aspp_outs = torch.cat([interpolate, x_141, x_144, x_147, x_150], dim=1)
        interpolate = x_141 = x_144 = x_147 = x_150 = None
        x_151 = torch.conv2d(
            aspp_outs,
            l_self_modules_decode_head_modules_bottleneck_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        aspp_outs = l_self_modules_decode_head_modules_bottleneck_modules_conv_parameters_weight_ = (None)
        x_152 = torch.nn.functional.batch_norm(
            x_151,
            l_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_151 = l_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_var_ = (
            l_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_weight_
        ) = (
            l_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_bias_
        ) = None
        x_153 = torch.nn.functional.relu(x_152, inplace=True)
        x_152 = None
        feat = torch.nn.functional.dropout2d(x_153, 0.1, False, False)
        x_153 = None
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
