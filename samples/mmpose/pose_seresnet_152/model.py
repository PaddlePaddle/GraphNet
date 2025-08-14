import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_inputs_: torch.Tensor,
        L_self_modules_backbone_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer1_modules_0_modules_se_layer_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_se_layer_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_se_layer_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_se_layer_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer1_modules_1_modules_se_layer_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_se_layer_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_se_layer_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_se_layer_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer1_modules_2_modules_se_layer_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_se_layer_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_se_layer_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_se_layer_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer2_modules_0_modules_se_layer_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_se_layer_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_se_layer_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_se_layer_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer2_modules_1_modules_se_layer_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_se_layer_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_se_layer_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_se_layer_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer2_modules_2_modules_se_layer_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_se_layer_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_se_layer_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_se_layer_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer2_modules_3_modules_se_layer_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_se_layer_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_se_layer_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_se_layer_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_4_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_4_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_4_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_4_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_4_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_4_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_4_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_4_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_4_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_4_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_4_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_4_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_4_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_4_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_4_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_4_modules_se_layer_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_4_modules_se_layer_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_4_modules_se_layer_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_4_modules_se_layer_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_5_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_5_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_5_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_5_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_5_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_5_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_5_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_5_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_5_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_5_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_5_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_5_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_5_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_5_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_5_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_5_modules_se_layer_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_5_modules_se_layer_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_5_modules_se_layer_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_5_modules_se_layer_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_6_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_6_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_6_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_6_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_6_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_6_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_6_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_6_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_6_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_6_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_6_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_6_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_6_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_6_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_6_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_6_modules_se_layer_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_6_modules_se_layer_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_6_modules_se_layer_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_6_modules_se_layer_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_7_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_7_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_7_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_7_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_7_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_7_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_7_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_7_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_7_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_7_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_7_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_7_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_7_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_7_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_7_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_7_modules_se_layer_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_7_modules_se_layer_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_7_modules_se_layer_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_7_modules_se_layer_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_0_modules_se_layer_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_se_layer_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_se_layer_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_se_layer_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_1_modules_se_layer_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_se_layer_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_se_layer_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_se_layer_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_2_modules_se_layer_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_se_layer_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_se_layer_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_se_layer_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_3_modules_se_layer_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_se_layer_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_se_layer_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_se_layer_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_4_modules_se_layer_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_se_layer_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_se_layer_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_se_layer_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_5_modules_se_layer_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_se_layer_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_se_layer_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_se_layer_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_6_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_6_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_6_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_6_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_6_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_6_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_se_layer_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_se_layer_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_se_layer_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_se_layer_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_7_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_7_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_7_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_7_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_7_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_7_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_7_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_7_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_7_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_7_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_7_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_7_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_7_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_7_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_7_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_7_modules_se_layer_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_7_modules_se_layer_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_7_modules_se_layer_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_7_modules_se_layer_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_8_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_8_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_8_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_8_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_8_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_8_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_8_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_8_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_8_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_8_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_8_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_8_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_8_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_8_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_8_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_8_modules_se_layer_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_8_modules_se_layer_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_8_modules_se_layer_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_8_modules_se_layer_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_9_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_9_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_9_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_9_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_9_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_9_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_9_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_9_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_9_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_9_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_9_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_9_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_9_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_9_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_9_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_9_modules_se_layer_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_9_modules_se_layer_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_9_modules_se_layer_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_9_modules_se_layer_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_10_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_10_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_10_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_10_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_10_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_10_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_10_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_10_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_10_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_10_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_10_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_10_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_10_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_10_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_10_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_10_modules_se_layer_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_10_modules_se_layer_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_10_modules_se_layer_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_10_modules_se_layer_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_11_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_11_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_11_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_11_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_11_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_11_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_11_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_11_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_11_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_11_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_11_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_11_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_11_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_11_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_11_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_11_modules_se_layer_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_11_modules_se_layer_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_11_modules_se_layer_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_11_modules_se_layer_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_12_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_12_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_12_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_12_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_12_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_12_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_12_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_12_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_12_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_12_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_12_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_12_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_12_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_12_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_12_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_12_modules_se_layer_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_12_modules_se_layer_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_12_modules_se_layer_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_12_modules_se_layer_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_13_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_13_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_13_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_13_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_13_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_13_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_13_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_13_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_13_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_13_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_13_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_13_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_13_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_13_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_13_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_13_modules_se_layer_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_13_modules_se_layer_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_13_modules_se_layer_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_13_modules_se_layer_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_14_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_14_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_14_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_14_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_14_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_14_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_14_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_14_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_14_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_14_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_14_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_14_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_14_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_14_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_14_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_14_modules_se_layer_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_14_modules_se_layer_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_14_modules_se_layer_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_14_modules_se_layer_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_15_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_15_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_15_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_15_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_15_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_15_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_15_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_15_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_15_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_15_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_15_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_15_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_15_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_15_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_15_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_15_modules_se_layer_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_15_modules_se_layer_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_15_modules_se_layer_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_15_modules_se_layer_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_16_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_16_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_16_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_16_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_16_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_16_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_16_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_16_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_16_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_16_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_16_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_16_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_16_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_16_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_16_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_16_modules_se_layer_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_16_modules_se_layer_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_16_modules_se_layer_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_16_modules_se_layer_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_17_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_17_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_17_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_17_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_17_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_17_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_17_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_17_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_17_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_17_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_17_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_17_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_17_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_17_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_17_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_17_modules_se_layer_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_17_modules_se_layer_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_17_modules_se_layer_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_17_modules_se_layer_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_18_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_18_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_18_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_18_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_18_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_18_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_18_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_18_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_18_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_18_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_18_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_18_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_18_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_18_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_18_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_18_modules_se_layer_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_18_modules_se_layer_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_18_modules_se_layer_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_18_modules_se_layer_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_19_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_19_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_19_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_19_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_19_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_19_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_19_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_19_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_19_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_19_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_19_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_19_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_19_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_19_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_19_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_19_modules_se_layer_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_19_modules_se_layer_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_19_modules_se_layer_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_19_modules_se_layer_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_20_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_20_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_20_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_20_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_20_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_20_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_20_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_20_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_20_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_20_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_20_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_20_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_20_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_20_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_20_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_20_modules_se_layer_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_20_modules_se_layer_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_20_modules_se_layer_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_20_modules_se_layer_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_21_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_21_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_21_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_21_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_21_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_21_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_21_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_21_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_21_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_21_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_21_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_21_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_21_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_21_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_21_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_21_modules_se_layer_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_21_modules_se_layer_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_21_modules_se_layer_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_21_modules_se_layer_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_22_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_22_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_22_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_22_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_22_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_22_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_22_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_22_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_22_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_22_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_22_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_22_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_22_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_22_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_22_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_22_modules_se_layer_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_22_modules_se_layer_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_22_modules_se_layer_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_22_modules_se_layer_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_23_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_23_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_23_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_23_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_23_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_23_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_23_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_23_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_23_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_23_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_23_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_23_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_23_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_23_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_23_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_23_modules_se_layer_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_23_modules_se_layer_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_23_modules_se_layer_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_23_modules_se_layer_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_24_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_24_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_24_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_24_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_24_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_24_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_24_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_24_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_24_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_24_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_24_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_24_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_24_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_24_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_24_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_24_modules_se_layer_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_24_modules_se_layer_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_24_modules_se_layer_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_24_modules_se_layer_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_25_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_25_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_25_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_25_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_25_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_25_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_25_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_25_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_25_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_25_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_25_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_25_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_25_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_25_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_25_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_25_modules_se_layer_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_25_modules_se_layer_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_25_modules_se_layer_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_25_modules_se_layer_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_26_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_26_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_26_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_26_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_26_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_26_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_26_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_26_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_26_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_26_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_26_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_26_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_26_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_26_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_26_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_26_modules_se_layer_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_26_modules_se_layer_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_26_modules_se_layer_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_26_modules_se_layer_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_27_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_27_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_27_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_27_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_27_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_27_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_27_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_27_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_27_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_27_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_27_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_27_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_27_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_27_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_27_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_27_modules_se_layer_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_27_modules_se_layer_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_27_modules_se_layer_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_27_modules_se_layer_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_28_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_28_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_28_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_28_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_28_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_28_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_28_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_28_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_28_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_28_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_28_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_28_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_28_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_28_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_28_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_28_modules_se_layer_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_28_modules_se_layer_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_28_modules_se_layer_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_28_modules_se_layer_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_29_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_29_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_29_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_29_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_29_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_29_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_29_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_29_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_29_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_29_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_29_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_29_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_29_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_29_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_29_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_29_modules_se_layer_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_29_modules_se_layer_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_29_modules_se_layer_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_29_modules_se_layer_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_30_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_30_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_30_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_30_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_30_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_30_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_30_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_30_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_30_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_30_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_30_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_30_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_30_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_30_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_30_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_30_modules_se_layer_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_30_modules_se_layer_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_30_modules_se_layer_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_30_modules_se_layer_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_31_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_31_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_31_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_31_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_31_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_31_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_31_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_31_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_31_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_31_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_31_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_31_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_31_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_31_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_31_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_31_modules_se_layer_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_31_modules_se_layer_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_31_modules_se_layer_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_31_modules_se_layer_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_32_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_32_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_32_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_32_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_32_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_32_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_32_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_32_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_32_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_32_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_32_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_32_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_32_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_32_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_32_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_32_modules_se_layer_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_32_modules_se_layer_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_32_modules_se_layer_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_32_modules_se_layer_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_33_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_33_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_33_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_33_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_33_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_33_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_33_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_33_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_33_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_33_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_33_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_33_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_33_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_33_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_33_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_33_modules_se_layer_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_33_modules_se_layer_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_33_modules_se_layer_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_33_modules_se_layer_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_34_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_34_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_34_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_34_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_34_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_34_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_34_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_34_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_34_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_34_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_34_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_34_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_34_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_34_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_34_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_34_modules_se_layer_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_34_modules_se_layer_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_34_modules_se_layer_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_34_modules_se_layer_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_35_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_35_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_35_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_35_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_35_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_35_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_35_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_35_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_35_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_35_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_35_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_35_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_35_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_35_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_35_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_35_modules_se_layer_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_35_modules_se_layer_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_35_modules_se_layer_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_35_modules_se_layer_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer4_modules_0_modules_se_layer_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_se_layer_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_se_layer_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_se_layer_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer4_modules_1_modules_se_layer_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_se_layer_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_se_layer_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_se_layer_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer4_modules_2_modules_se_layer_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_se_layer_modules_conv1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_se_layer_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_se_layer_modules_conv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_backbone_modules_conv1_parameters_weight_ = (
            L_self_modules_backbone_modules_conv1_parameters_weight_
        )
        l_self_modules_backbone_modules_bn1_buffers_running_mean_ = (
            L_self_modules_backbone_modules_bn1_buffers_running_mean_
        )
        l_self_modules_backbone_modules_bn1_buffers_running_var_ = (
            L_self_modules_backbone_modules_bn1_buffers_running_var_
        )
        l_self_modules_backbone_modules_bn1_parameters_weight_ = (
            L_self_modules_backbone_modules_bn1_parameters_weight_
        )
        l_self_modules_backbone_modules_bn1_parameters_bias_ = (
            L_self_modules_backbone_modules_bn1_parameters_bias_
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
        l_self_modules_backbone_modules_layer1_modules_0_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_se_layer_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_0_modules_se_layer_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_0_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_se_layer_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_0_modules_se_layer_modules_conv2_modules_conv_parameters_bias_
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
        l_self_modules_backbone_modules_layer1_modules_1_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_1_modules_se_layer_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_1_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_1_modules_se_layer_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_1_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_1_modules_se_layer_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_1_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_1_modules_se_layer_modules_conv2_modules_conv_parameters_bias_
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
        l_self_modules_backbone_modules_layer1_modules_2_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_2_modules_se_layer_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_2_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_2_modules_se_layer_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_2_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_2_modules_se_layer_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_2_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_2_modules_se_layer_modules_conv2_modules_conv_parameters_bias_
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
        l_self_modules_backbone_modules_layer2_modules_0_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_se_layer_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_0_modules_se_layer_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_0_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_se_layer_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_0_modules_se_layer_modules_conv2_modules_conv_parameters_bias_
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
        l_self_modules_backbone_modules_layer2_modules_1_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_se_layer_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_1_modules_se_layer_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_1_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_se_layer_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_1_modules_se_layer_modules_conv2_modules_conv_parameters_bias_
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
        l_self_modules_backbone_modules_layer2_modules_2_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_2_modules_se_layer_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_2_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_2_modules_se_layer_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_2_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_2_modules_se_layer_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_2_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_2_modules_se_layer_modules_conv2_modules_conv_parameters_bias_
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
        l_self_modules_backbone_modules_layer2_modules_3_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_3_modules_se_layer_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_3_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_3_modules_se_layer_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_3_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_3_modules_se_layer_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_3_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_3_modules_se_layer_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_4_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_4_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_4_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_4_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_4_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_4_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_4_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_4_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_4_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_4_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_4_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_4_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_4_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_4_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_4_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_4_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_4_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_4_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_4_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_4_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_4_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_4_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_4_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_4_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_4_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_4_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_4_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_4_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_4_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_4_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_4_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_4_modules_se_layer_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_4_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_4_modules_se_layer_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_4_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_4_modules_se_layer_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_4_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_4_modules_se_layer_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_5_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_5_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_5_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_5_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_5_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_5_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_5_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_5_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_5_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_5_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_5_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_5_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_5_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_5_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_5_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_5_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_5_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_5_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_5_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_5_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_5_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_5_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_5_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_5_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_5_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_5_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_5_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_5_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_5_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_5_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_5_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_5_modules_se_layer_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_5_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_5_modules_se_layer_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_5_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_5_modules_se_layer_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_5_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_5_modules_se_layer_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_6_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_6_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_6_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_6_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_6_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_6_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_6_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_6_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_6_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_6_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_6_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_6_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_6_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_6_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_6_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_6_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_6_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_6_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_6_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_6_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_6_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_6_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_6_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_6_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_6_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_6_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_6_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_6_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_6_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_6_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_6_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_6_modules_se_layer_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_6_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_6_modules_se_layer_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_6_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_6_modules_se_layer_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_6_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_6_modules_se_layer_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_7_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_7_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_7_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_7_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_7_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_7_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_7_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_7_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_7_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_7_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_7_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_7_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_7_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_7_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_7_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_7_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_7_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_7_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_7_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_7_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_7_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_7_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_7_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_7_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_7_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_7_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_7_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_7_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_7_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_7_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_7_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_7_modules_se_layer_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_7_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_7_modules_se_layer_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_7_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_7_modules_se_layer_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_7_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_7_modules_se_layer_modules_conv2_modules_conv_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_0_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_se_layer_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_0_modules_se_layer_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_0_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_se_layer_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_0_modules_se_layer_modules_conv2_modules_conv_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_1_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_se_layer_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_1_modules_se_layer_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_1_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_se_layer_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_1_modules_se_layer_modules_conv2_modules_conv_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_2_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_se_layer_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_2_modules_se_layer_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_2_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_se_layer_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_2_modules_se_layer_modules_conv2_modules_conv_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_3_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_3_modules_se_layer_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_3_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_3_modules_se_layer_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_3_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_3_modules_se_layer_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_3_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_3_modules_se_layer_modules_conv2_modules_conv_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_4_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_4_modules_se_layer_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_4_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_4_modules_se_layer_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_4_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_4_modules_se_layer_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_4_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_4_modules_se_layer_modules_conv2_modules_conv_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_5_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_5_modules_se_layer_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_5_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_5_modules_se_layer_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_5_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_5_modules_se_layer_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_5_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_5_modules_se_layer_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_6_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_6_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_6_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_6_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_6_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_6_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_6_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_6_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_6_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_6_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_6_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_6_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_6_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_6_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_6_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_6_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_6_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_6_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_6_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_6_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_6_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_6_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_6_modules_se_layer_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_6_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_6_modules_se_layer_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_6_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_6_modules_se_layer_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_6_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_6_modules_se_layer_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_7_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_7_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_7_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_7_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_7_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_7_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_7_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_7_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_7_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_7_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_7_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_7_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_7_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_7_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_7_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_7_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_7_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_7_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_7_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_7_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_7_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_7_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_7_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_7_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_7_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_7_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_7_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_7_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_7_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_7_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_7_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_7_modules_se_layer_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_7_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_7_modules_se_layer_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_7_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_7_modules_se_layer_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_7_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_7_modules_se_layer_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_8_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_8_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_8_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_8_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_8_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_8_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_8_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_8_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_8_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_8_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_8_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_8_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_8_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_8_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_8_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_8_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_8_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_8_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_8_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_8_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_8_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_8_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_8_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_8_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_8_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_8_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_8_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_8_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_8_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_8_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_8_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_8_modules_se_layer_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_8_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_8_modules_se_layer_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_8_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_8_modules_se_layer_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_8_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_8_modules_se_layer_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_9_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_9_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_9_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_9_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_9_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_9_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_9_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_9_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_9_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_9_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_9_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_9_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_9_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_9_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_9_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_9_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_9_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_9_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_9_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_9_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_9_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_9_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_9_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_9_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_9_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_9_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_9_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_9_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_9_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_9_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_9_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_9_modules_se_layer_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_9_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_9_modules_se_layer_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_9_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_9_modules_se_layer_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_9_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_9_modules_se_layer_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_10_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_10_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_10_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_10_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_10_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_10_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_10_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_10_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_10_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_10_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_10_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_10_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_10_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_10_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_10_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_10_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_10_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_10_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_10_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_10_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_10_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_10_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_10_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_10_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_10_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_10_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_10_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_10_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_10_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_10_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_10_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_10_modules_se_layer_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_10_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_10_modules_se_layer_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_10_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_10_modules_se_layer_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_10_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_10_modules_se_layer_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_11_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_11_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_11_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_11_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_11_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_11_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_11_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_11_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_11_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_11_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_11_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_11_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_11_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_11_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_11_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_11_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_11_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_11_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_11_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_11_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_11_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_11_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_11_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_11_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_11_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_11_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_11_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_11_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_11_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_11_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_11_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_11_modules_se_layer_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_11_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_11_modules_se_layer_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_11_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_11_modules_se_layer_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_11_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_11_modules_se_layer_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_12_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_12_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_12_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_12_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_12_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_12_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_12_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_12_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_12_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_12_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_12_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_12_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_12_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_12_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_12_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_12_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_12_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_12_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_12_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_12_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_12_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_12_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_12_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_12_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_12_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_12_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_12_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_12_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_12_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_12_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_12_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_12_modules_se_layer_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_12_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_12_modules_se_layer_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_12_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_12_modules_se_layer_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_12_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_12_modules_se_layer_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_13_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_13_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_13_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_13_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_13_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_13_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_13_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_13_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_13_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_13_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_13_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_13_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_13_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_13_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_13_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_13_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_13_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_13_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_13_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_13_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_13_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_13_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_13_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_13_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_13_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_13_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_13_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_13_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_13_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_13_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_13_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_13_modules_se_layer_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_13_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_13_modules_se_layer_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_13_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_13_modules_se_layer_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_13_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_13_modules_se_layer_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_14_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_14_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_14_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_14_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_14_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_14_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_14_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_14_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_14_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_14_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_14_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_14_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_14_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_14_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_14_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_14_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_14_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_14_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_14_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_14_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_14_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_14_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_14_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_14_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_14_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_14_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_14_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_14_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_14_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_14_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_14_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_14_modules_se_layer_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_14_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_14_modules_se_layer_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_14_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_14_modules_se_layer_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_14_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_14_modules_se_layer_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_15_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_15_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_15_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_15_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_15_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_15_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_15_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_15_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_15_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_15_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_15_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_15_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_15_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_15_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_15_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_15_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_15_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_15_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_15_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_15_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_15_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_15_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_15_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_15_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_15_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_15_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_15_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_15_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_15_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_15_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_15_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_15_modules_se_layer_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_15_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_15_modules_se_layer_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_15_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_15_modules_se_layer_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_15_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_15_modules_se_layer_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_16_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_16_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_16_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_16_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_16_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_16_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_16_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_16_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_16_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_16_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_16_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_16_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_16_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_16_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_16_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_16_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_16_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_16_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_16_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_16_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_16_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_16_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_16_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_16_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_16_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_16_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_16_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_16_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_16_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_16_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_16_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_16_modules_se_layer_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_16_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_16_modules_se_layer_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_16_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_16_modules_se_layer_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_16_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_16_modules_se_layer_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_17_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_17_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_17_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_17_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_17_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_17_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_17_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_17_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_17_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_17_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_17_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_17_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_17_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_17_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_17_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_17_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_17_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_17_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_17_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_17_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_17_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_17_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_17_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_17_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_17_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_17_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_17_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_17_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_17_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_17_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_17_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_17_modules_se_layer_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_17_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_17_modules_se_layer_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_17_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_17_modules_se_layer_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_17_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_17_modules_se_layer_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_18_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_18_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_18_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_18_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_18_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_18_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_18_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_18_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_18_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_18_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_18_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_18_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_18_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_18_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_18_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_18_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_18_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_18_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_18_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_18_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_18_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_18_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_18_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_18_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_18_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_18_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_18_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_18_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_18_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_18_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_18_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_18_modules_se_layer_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_18_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_18_modules_se_layer_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_18_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_18_modules_se_layer_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_18_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_18_modules_se_layer_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_19_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_19_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_19_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_19_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_19_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_19_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_19_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_19_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_19_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_19_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_19_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_19_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_19_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_19_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_19_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_19_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_19_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_19_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_19_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_19_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_19_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_19_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_19_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_19_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_19_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_19_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_19_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_19_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_19_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_19_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_19_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_19_modules_se_layer_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_19_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_19_modules_se_layer_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_19_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_19_modules_se_layer_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_19_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_19_modules_se_layer_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_20_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_20_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_20_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_20_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_20_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_20_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_20_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_20_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_20_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_20_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_20_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_20_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_20_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_20_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_20_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_20_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_20_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_20_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_20_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_20_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_20_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_20_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_20_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_20_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_20_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_20_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_20_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_20_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_20_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_20_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_20_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_20_modules_se_layer_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_20_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_20_modules_se_layer_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_20_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_20_modules_se_layer_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_20_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_20_modules_se_layer_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_21_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_21_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_21_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_21_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_21_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_21_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_21_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_21_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_21_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_21_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_21_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_21_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_21_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_21_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_21_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_21_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_21_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_21_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_21_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_21_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_21_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_21_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_21_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_21_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_21_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_21_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_21_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_21_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_21_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_21_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_21_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_21_modules_se_layer_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_21_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_21_modules_se_layer_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_21_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_21_modules_se_layer_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_21_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_21_modules_se_layer_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_22_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_22_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_22_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_22_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_22_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_22_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_22_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_22_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_22_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_22_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_22_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_22_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_22_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_22_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_22_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_22_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_22_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_22_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_22_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_22_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_22_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_22_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_22_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_22_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_22_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_22_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_22_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_22_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_22_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_22_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_22_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_22_modules_se_layer_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_22_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_22_modules_se_layer_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_22_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_22_modules_se_layer_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_22_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_22_modules_se_layer_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_23_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_23_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_23_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_23_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_23_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_23_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_23_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_23_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_23_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_23_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_23_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_23_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_23_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_23_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_23_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_23_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_23_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_23_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_23_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_23_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_23_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_23_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_23_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_23_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_23_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_23_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_23_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_23_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_23_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_23_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_23_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_23_modules_se_layer_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_23_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_23_modules_se_layer_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_23_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_23_modules_se_layer_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_23_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_23_modules_se_layer_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_24_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_24_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_24_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_24_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_24_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_24_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_24_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_24_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_24_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_24_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_24_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_24_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_24_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_24_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_24_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_24_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_24_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_24_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_24_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_24_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_24_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_24_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_24_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_24_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_24_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_24_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_24_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_24_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_24_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_24_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_24_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_24_modules_se_layer_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_24_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_24_modules_se_layer_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_24_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_24_modules_se_layer_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_24_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_24_modules_se_layer_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_25_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_25_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_25_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_25_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_25_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_25_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_25_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_25_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_25_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_25_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_25_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_25_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_25_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_25_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_25_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_25_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_25_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_25_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_25_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_25_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_25_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_25_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_25_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_25_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_25_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_25_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_25_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_25_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_25_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_25_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_25_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_25_modules_se_layer_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_25_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_25_modules_se_layer_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_25_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_25_modules_se_layer_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_25_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_25_modules_se_layer_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_26_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_26_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_26_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_26_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_26_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_26_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_26_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_26_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_26_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_26_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_26_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_26_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_26_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_26_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_26_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_26_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_26_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_26_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_26_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_26_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_26_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_26_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_26_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_26_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_26_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_26_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_26_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_26_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_26_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_26_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_26_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_26_modules_se_layer_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_26_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_26_modules_se_layer_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_26_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_26_modules_se_layer_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_26_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_26_modules_se_layer_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_27_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_27_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_27_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_27_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_27_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_27_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_27_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_27_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_27_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_27_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_27_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_27_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_27_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_27_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_27_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_27_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_27_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_27_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_27_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_27_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_27_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_27_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_27_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_27_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_27_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_27_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_27_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_27_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_27_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_27_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_27_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_27_modules_se_layer_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_27_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_27_modules_se_layer_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_27_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_27_modules_se_layer_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_27_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_27_modules_se_layer_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_28_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_28_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_28_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_28_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_28_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_28_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_28_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_28_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_28_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_28_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_28_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_28_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_28_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_28_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_28_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_28_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_28_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_28_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_28_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_28_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_28_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_28_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_28_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_28_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_28_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_28_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_28_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_28_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_28_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_28_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_28_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_28_modules_se_layer_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_28_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_28_modules_se_layer_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_28_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_28_modules_se_layer_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_28_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_28_modules_se_layer_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_29_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_29_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_29_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_29_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_29_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_29_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_29_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_29_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_29_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_29_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_29_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_29_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_29_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_29_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_29_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_29_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_29_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_29_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_29_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_29_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_29_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_29_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_29_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_29_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_29_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_29_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_29_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_29_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_29_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_29_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_29_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_29_modules_se_layer_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_29_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_29_modules_se_layer_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_29_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_29_modules_se_layer_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_29_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_29_modules_se_layer_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_30_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_30_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_30_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_30_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_30_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_30_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_30_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_30_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_30_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_30_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_30_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_30_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_30_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_30_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_30_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_30_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_30_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_30_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_30_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_30_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_30_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_30_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_30_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_30_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_30_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_30_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_30_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_30_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_30_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_30_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_30_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_30_modules_se_layer_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_30_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_30_modules_se_layer_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_30_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_30_modules_se_layer_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_30_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_30_modules_se_layer_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_31_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_31_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_31_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_31_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_31_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_31_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_31_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_31_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_31_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_31_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_31_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_31_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_31_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_31_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_31_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_31_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_31_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_31_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_31_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_31_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_31_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_31_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_31_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_31_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_31_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_31_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_31_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_31_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_31_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_31_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_31_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_31_modules_se_layer_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_31_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_31_modules_se_layer_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_31_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_31_modules_se_layer_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_31_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_31_modules_se_layer_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_32_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_32_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_32_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_32_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_32_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_32_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_32_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_32_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_32_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_32_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_32_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_32_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_32_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_32_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_32_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_32_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_32_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_32_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_32_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_32_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_32_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_32_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_32_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_32_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_32_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_32_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_32_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_32_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_32_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_32_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_32_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_32_modules_se_layer_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_32_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_32_modules_se_layer_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_32_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_32_modules_se_layer_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_32_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_32_modules_se_layer_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_33_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_33_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_33_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_33_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_33_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_33_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_33_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_33_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_33_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_33_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_33_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_33_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_33_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_33_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_33_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_33_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_33_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_33_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_33_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_33_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_33_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_33_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_33_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_33_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_33_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_33_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_33_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_33_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_33_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_33_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_33_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_33_modules_se_layer_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_33_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_33_modules_se_layer_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_33_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_33_modules_se_layer_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_33_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_33_modules_se_layer_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_34_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_34_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_34_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_34_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_34_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_34_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_34_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_34_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_34_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_34_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_34_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_34_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_34_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_34_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_34_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_34_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_34_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_34_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_34_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_34_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_34_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_34_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_34_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_34_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_34_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_34_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_34_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_34_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_34_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_34_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_34_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_34_modules_se_layer_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_34_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_34_modules_se_layer_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_34_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_34_modules_se_layer_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_34_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_34_modules_se_layer_modules_conv2_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_35_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_35_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_35_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_35_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_35_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_35_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_35_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_35_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_35_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_35_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_35_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_35_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_35_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_35_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_35_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_35_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_35_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_35_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_35_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_35_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_35_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_35_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_35_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_35_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_35_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_35_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_35_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_35_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_35_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_35_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_35_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_35_modules_se_layer_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_35_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_35_modules_se_layer_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_35_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_35_modules_se_layer_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_35_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_35_modules_se_layer_modules_conv2_modules_conv_parameters_bias_
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
        l_self_modules_backbone_modules_layer4_modules_0_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_se_layer_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_0_modules_se_layer_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_0_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_se_layer_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_0_modules_se_layer_modules_conv2_modules_conv_parameters_bias_
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
        l_self_modules_backbone_modules_layer4_modules_1_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_se_layer_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_1_modules_se_layer_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_1_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_se_layer_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_1_modules_se_layer_modules_conv2_modules_conv_parameters_bias_
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
        l_self_modules_backbone_modules_layer4_modules_2_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_se_layer_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_2_modules_se_layer_modules_conv1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_2_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_se_layer_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_2_modules_se_layer_modules_conv2_modules_conv_parameters_bias_
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
            l_self_modules_backbone_modules_conv1_parameters_weight_,
            None,
            (2, 2),
            (3, 3),
            (1, 1),
            1,
        )
        l_inputs_ = l_self_modules_backbone_modules_conv1_parameters_weight_ = None
        x_1 = torch.nn.functional.batch_norm(
            x,
            l_self_modules_backbone_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x = (
            l_self_modules_backbone_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_backbone_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_backbone_modules_bn1_parameters_weight_
        ) = l_self_modules_backbone_modules_bn1_parameters_bias_ = None
        x_2 = torch.nn.functional.relu(x_1, inplace=True)
        x_1 = None
        x_3 = torch.nn.functional.max_pool2d(
            x_2, 3, 2, 1, 1, ceil_mode=False, return_indices=False
        )
        x_2 = None
        out = torch.conv2d(
            x_3,
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
        out_8 = torch.nn.functional.adaptive_avg_pool2d(out_7, 1)
        x_4 = torch.conv2d(
            out_8,
            l_self_modules_backbone_modules_layer1_modules_0_modules_se_layer_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_se_layer_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_8 = l_self_modules_backbone_modules_layer1_modules_0_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_0_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_5 = torch.nn.functional.relu(x_4, inplace=True)
        x_4 = None
        x_6 = torch.conv2d(
            x_5,
            l_self_modules_backbone_modules_layer1_modules_0_modules_se_layer_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_se_layer_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_5 = l_self_modules_backbone_modules_layer1_modules_0_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_0_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = (None)
        x_7 = torch.sigmoid(x_6)
        x_6 = None
        out_9 = out_7 * x_7
        out_7 = x_7 = None
        input_1 = torch.conv2d(
            x_3,
            l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_3 = l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
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
        out_9 += input_2
        out_10 = out_9
        out_9 = input_2 = None
        out_11 = torch.nn.functional.relu(out_10, inplace=True)
        out_10 = None
        out_12 = torch.conv2d(
            out_11,
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
        out_13 = torch.nn.functional.batch_norm(
            out_12,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_12 = l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_parameters_bias_ = (None)
        out_14 = torch.nn.functional.relu(out_13, inplace=True)
        out_13 = None
        out_15 = torch.conv2d(
            out_14,
            l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_14 = l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_parameters_weight_ = (None)
        out_16 = torch.nn.functional.batch_norm(
            out_15,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_15 = l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_parameters_bias_ = (None)
        out_17 = torch.nn.functional.relu(out_16, inplace=True)
        out_16 = None
        out_18 = torch.conv2d(
            out_17,
            l_self_modules_backbone_modules_layer1_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_17 = l_self_modules_backbone_modules_layer1_modules_1_modules_conv3_parameters_weight_ = (None)
        out_19 = torch.nn.functional.batch_norm(
            out_18,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_18 = l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_parameters_bias_ = (None)
        out_20 = torch.nn.functional.adaptive_avg_pool2d(out_19, 1)
        x_8 = torch.conv2d(
            out_20,
            l_self_modules_backbone_modules_layer1_modules_1_modules_se_layer_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_se_layer_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_20 = l_self_modules_backbone_modules_layer1_modules_1_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_1_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_9 = torch.nn.functional.relu(x_8, inplace=True)
        x_8 = None
        x_10 = torch.conv2d(
            x_9,
            l_self_modules_backbone_modules_layer1_modules_1_modules_se_layer_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_se_layer_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_9 = l_self_modules_backbone_modules_layer1_modules_1_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_1_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = (None)
        x_11 = torch.sigmoid(x_10)
        x_10 = None
        out_21 = out_19 * x_11
        out_19 = x_11 = None
        out_21 += out_11
        out_22 = out_21
        out_21 = out_11 = None
        out_23 = torch.nn.functional.relu(out_22, inplace=True)
        out_22 = None
        out_24 = torch.conv2d(
            out_23,
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
        out_25 = torch.nn.functional.batch_norm(
            out_24,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_24 = l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_parameters_bias_ = (None)
        out_26 = torch.nn.functional.relu(out_25, inplace=True)
        out_25 = None
        out_27 = torch.conv2d(
            out_26,
            l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_26 = l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_parameters_weight_ = (None)
        out_28 = torch.nn.functional.batch_norm(
            out_27,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_27 = l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_parameters_bias_ = (None)
        out_29 = torch.nn.functional.relu(out_28, inplace=True)
        out_28 = None
        out_30 = torch.conv2d(
            out_29,
            l_self_modules_backbone_modules_layer1_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_29 = l_self_modules_backbone_modules_layer1_modules_2_modules_conv3_parameters_weight_ = (None)
        out_31 = torch.nn.functional.batch_norm(
            out_30,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_30 = l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_parameters_bias_ = (None)
        out_32 = torch.nn.functional.adaptive_avg_pool2d(out_31, 1)
        x_12 = torch.conv2d(
            out_32,
            l_self_modules_backbone_modules_layer1_modules_2_modules_se_layer_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_se_layer_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_32 = l_self_modules_backbone_modules_layer1_modules_2_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_2_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_13 = torch.nn.functional.relu(x_12, inplace=True)
        x_12 = None
        x_14 = torch.conv2d(
            x_13,
            l_self_modules_backbone_modules_layer1_modules_2_modules_se_layer_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_se_layer_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_13 = l_self_modules_backbone_modules_layer1_modules_2_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_2_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = (None)
        x_15 = torch.sigmoid(x_14)
        x_14 = None
        out_33 = out_31 * x_15
        out_31 = x_15 = None
        out_33 += out_23
        out_34 = out_33
        out_33 = out_23 = None
        out_35 = torch.nn.functional.relu(out_34, inplace=True)
        out_34 = None
        out_36 = torch.conv2d(
            out_35,
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
        out_37 = torch.nn.functional.batch_norm(
            out_36,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_36 = l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_parameters_bias_ = (None)
        out_38 = torch.nn.functional.relu(out_37, inplace=True)
        out_37 = None
        out_39 = torch.conv2d(
            out_38,
            l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        out_38 = l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_parameters_weight_ = (None)
        out_40 = torch.nn.functional.batch_norm(
            out_39,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_39 = l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_parameters_bias_ = (None)
        out_41 = torch.nn.functional.relu(out_40, inplace=True)
        out_40 = None
        out_42 = torch.conv2d(
            out_41,
            l_self_modules_backbone_modules_layer2_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_41 = l_self_modules_backbone_modules_layer2_modules_0_modules_conv3_parameters_weight_ = (None)
        out_43 = torch.nn.functional.batch_norm(
            out_42,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_42 = l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_bias_ = (None)
        out_44 = torch.nn.functional.adaptive_avg_pool2d(out_43, 1)
        x_16 = torch.conv2d(
            out_44,
            l_self_modules_backbone_modules_layer2_modules_0_modules_se_layer_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_se_layer_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_44 = l_self_modules_backbone_modules_layer2_modules_0_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_0_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_17 = torch.nn.functional.relu(x_16, inplace=True)
        x_16 = None
        x_18 = torch.conv2d(
            x_17,
            l_self_modules_backbone_modules_layer2_modules_0_modules_se_layer_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_se_layer_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_17 = l_self_modules_backbone_modules_layer2_modules_0_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_0_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = (None)
        x_19 = torch.sigmoid(x_18)
        x_18 = None
        out_45 = out_43 * x_19
        out_43 = x_19 = None
        input_3 = torch.conv2d(
            out_35,
            l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        out_35 = l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_4 = torch.nn.functional.batch_norm(
            input_3,
            l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_3 = l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        out_45 += input_4
        out_46 = out_45
        out_45 = input_4 = None
        out_47 = torch.nn.functional.relu(out_46, inplace=True)
        out_46 = None
        out_48 = torch.conv2d(
            out_47,
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
        out_49 = torch.nn.functional.batch_norm(
            out_48,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_48 = l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_bias_ = (None)
        out_50 = torch.nn.functional.relu(out_49, inplace=True)
        out_49 = None
        out_51 = torch.conv2d(
            out_50,
            l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_50 = l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_parameters_weight_ = (None)
        out_52 = torch.nn.functional.batch_norm(
            out_51,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_51 = l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_parameters_bias_ = (None)
        out_53 = torch.nn.functional.relu(out_52, inplace=True)
        out_52 = None
        out_54 = torch.conv2d(
            out_53,
            l_self_modules_backbone_modules_layer2_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_53 = l_self_modules_backbone_modules_layer2_modules_1_modules_conv3_parameters_weight_ = (None)
        out_55 = torch.nn.functional.batch_norm(
            out_54,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_54 = l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_parameters_bias_ = (None)
        out_56 = torch.nn.functional.adaptive_avg_pool2d(out_55, 1)
        x_20 = torch.conv2d(
            out_56,
            l_self_modules_backbone_modules_layer2_modules_1_modules_se_layer_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_se_layer_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_56 = l_self_modules_backbone_modules_layer2_modules_1_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_1_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_21 = torch.nn.functional.relu(x_20, inplace=True)
        x_20 = None
        x_22 = torch.conv2d(
            x_21,
            l_self_modules_backbone_modules_layer2_modules_1_modules_se_layer_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_se_layer_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_21 = l_self_modules_backbone_modules_layer2_modules_1_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_1_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = (None)
        x_23 = torch.sigmoid(x_22)
        x_22 = None
        out_57 = out_55 * x_23
        out_55 = x_23 = None
        out_57 += out_47
        out_58 = out_57
        out_57 = out_47 = None
        out_59 = torch.nn.functional.relu(out_58, inplace=True)
        out_58 = None
        out_60 = torch.conv2d(
            out_59,
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
        out_61 = torch.nn.functional.batch_norm(
            out_60,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_60 = l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_parameters_bias_ = (None)
        out_62 = torch.nn.functional.relu(out_61, inplace=True)
        out_61 = None
        out_63 = torch.conv2d(
            out_62,
            l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_62 = l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_parameters_weight_ = (None)
        out_64 = torch.nn.functional.batch_norm(
            out_63,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_63 = l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_parameters_bias_ = (None)
        out_65 = torch.nn.functional.relu(out_64, inplace=True)
        out_64 = None
        out_66 = torch.conv2d(
            out_65,
            l_self_modules_backbone_modules_layer2_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_65 = l_self_modules_backbone_modules_layer2_modules_2_modules_conv3_parameters_weight_ = (None)
        out_67 = torch.nn.functional.batch_norm(
            out_66,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_66 = l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_parameters_bias_ = (None)
        out_68 = torch.nn.functional.adaptive_avg_pool2d(out_67, 1)
        x_24 = torch.conv2d(
            out_68,
            l_self_modules_backbone_modules_layer2_modules_2_modules_se_layer_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_se_layer_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_68 = l_self_modules_backbone_modules_layer2_modules_2_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_2_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_25 = torch.nn.functional.relu(x_24, inplace=True)
        x_24 = None
        x_26 = torch.conv2d(
            x_25,
            l_self_modules_backbone_modules_layer2_modules_2_modules_se_layer_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_se_layer_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_25 = l_self_modules_backbone_modules_layer2_modules_2_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_2_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = (None)
        x_27 = torch.sigmoid(x_26)
        x_26 = None
        out_69 = out_67 * x_27
        out_67 = x_27 = None
        out_69 += out_59
        out_70 = out_69
        out_69 = out_59 = None
        out_71 = torch.nn.functional.relu(out_70, inplace=True)
        out_70 = None
        out_72 = torch.conv2d(
            out_71,
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
        out_73 = torch.nn.functional.batch_norm(
            out_72,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_72 = l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_parameters_bias_ = (None)
        out_74 = torch.nn.functional.relu(out_73, inplace=True)
        out_73 = None
        out_75 = torch.conv2d(
            out_74,
            l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_74 = l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_parameters_weight_ = (None)
        out_76 = torch.nn.functional.batch_norm(
            out_75,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_75 = l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_parameters_bias_ = (None)
        out_77 = torch.nn.functional.relu(out_76, inplace=True)
        out_76 = None
        out_78 = torch.conv2d(
            out_77,
            l_self_modules_backbone_modules_layer2_modules_3_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_77 = l_self_modules_backbone_modules_layer2_modules_3_modules_conv3_parameters_weight_ = (None)
        out_79 = torch.nn.functional.batch_norm(
            out_78,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_78 = l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_parameters_bias_ = (None)
        out_80 = torch.nn.functional.adaptive_avg_pool2d(out_79, 1)
        x_28 = torch.conv2d(
            out_80,
            l_self_modules_backbone_modules_layer2_modules_3_modules_se_layer_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_se_layer_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_80 = l_self_modules_backbone_modules_layer2_modules_3_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_3_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_29 = torch.nn.functional.relu(x_28, inplace=True)
        x_28 = None
        x_30 = torch.conv2d(
            x_29,
            l_self_modules_backbone_modules_layer2_modules_3_modules_se_layer_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_se_layer_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_29 = l_self_modules_backbone_modules_layer2_modules_3_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_3_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = (None)
        x_31 = torch.sigmoid(x_30)
        x_30 = None
        out_81 = out_79 * x_31
        out_79 = x_31 = None
        out_81 += out_71
        out_82 = out_81
        out_81 = out_71 = None
        out_83 = torch.nn.functional.relu(out_82, inplace=True)
        out_82 = None
        out_84 = torch.conv2d(
            out_83,
            l_self_modules_backbone_modules_layer2_modules_4_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer2_modules_4_modules_conv1_parameters_weight_ = (
            None
        )
        out_85 = torch.nn.functional.batch_norm(
            out_84,
            l_self_modules_backbone_modules_layer2_modules_4_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_4_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_4_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_4_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_84 = l_self_modules_backbone_modules_layer2_modules_4_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_4_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_4_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_4_modules_bn1_parameters_bias_ = (None)
        out_86 = torch.nn.functional.relu(out_85, inplace=True)
        out_85 = None
        out_87 = torch.conv2d(
            out_86,
            l_self_modules_backbone_modules_layer2_modules_4_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_86 = l_self_modules_backbone_modules_layer2_modules_4_modules_conv2_parameters_weight_ = (None)
        out_88 = torch.nn.functional.batch_norm(
            out_87,
            l_self_modules_backbone_modules_layer2_modules_4_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_4_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_4_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_4_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_87 = l_self_modules_backbone_modules_layer2_modules_4_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_4_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_4_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_4_modules_bn2_parameters_bias_ = (None)
        out_89 = torch.nn.functional.relu(out_88, inplace=True)
        out_88 = None
        out_90 = torch.conv2d(
            out_89,
            l_self_modules_backbone_modules_layer2_modules_4_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_89 = l_self_modules_backbone_modules_layer2_modules_4_modules_conv3_parameters_weight_ = (None)
        out_91 = torch.nn.functional.batch_norm(
            out_90,
            l_self_modules_backbone_modules_layer2_modules_4_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_4_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_4_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_4_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_90 = l_self_modules_backbone_modules_layer2_modules_4_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_4_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_4_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_4_modules_bn3_parameters_bias_ = (None)
        out_92 = torch.nn.functional.adaptive_avg_pool2d(out_91, 1)
        x_32 = torch.conv2d(
            out_92,
            l_self_modules_backbone_modules_layer2_modules_4_modules_se_layer_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_4_modules_se_layer_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_92 = l_self_modules_backbone_modules_layer2_modules_4_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_4_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_33 = torch.nn.functional.relu(x_32, inplace=True)
        x_32 = None
        x_34 = torch.conv2d(
            x_33,
            l_self_modules_backbone_modules_layer2_modules_4_modules_se_layer_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_4_modules_se_layer_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_33 = l_self_modules_backbone_modules_layer2_modules_4_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_4_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = (None)
        x_35 = torch.sigmoid(x_34)
        x_34 = None
        out_93 = out_91 * x_35
        out_91 = x_35 = None
        out_93 += out_83
        out_94 = out_93
        out_93 = out_83 = None
        out_95 = torch.nn.functional.relu(out_94, inplace=True)
        out_94 = None
        out_96 = torch.conv2d(
            out_95,
            l_self_modules_backbone_modules_layer2_modules_5_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer2_modules_5_modules_conv1_parameters_weight_ = (
            None
        )
        out_97 = torch.nn.functional.batch_norm(
            out_96,
            l_self_modules_backbone_modules_layer2_modules_5_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_5_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_5_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_5_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_96 = l_self_modules_backbone_modules_layer2_modules_5_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_5_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_5_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_5_modules_bn1_parameters_bias_ = (None)
        out_98 = torch.nn.functional.relu(out_97, inplace=True)
        out_97 = None
        out_99 = torch.conv2d(
            out_98,
            l_self_modules_backbone_modules_layer2_modules_5_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_98 = l_self_modules_backbone_modules_layer2_modules_5_modules_conv2_parameters_weight_ = (None)
        out_100 = torch.nn.functional.batch_norm(
            out_99,
            l_self_modules_backbone_modules_layer2_modules_5_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_5_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_5_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_5_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_99 = l_self_modules_backbone_modules_layer2_modules_5_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_5_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_5_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_5_modules_bn2_parameters_bias_ = (None)
        out_101 = torch.nn.functional.relu(out_100, inplace=True)
        out_100 = None
        out_102 = torch.conv2d(
            out_101,
            l_self_modules_backbone_modules_layer2_modules_5_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_101 = l_self_modules_backbone_modules_layer2_modules_5_modules_conv3_parameters_weight_ = (None)
        out_103 = torch.nn.functional.batch_norm(
            out_102,
            l_self_modules_backbone_modules_layer2_modules_5_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_5_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_5_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_5_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_102 = l_self_modules_backbone_modules_layer2_modules_5_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_5_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_5_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_5_modules_bn3_parameters_bias_ = (None)
        out_104 = torch.nn.functional.adaptive_avg_pool2d(out_103, 1)
        x_36 = torch.conv2d(
            out_104,
            l_self_modules_backbone_modules_layer2_modules_5_modules_se_layer_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_5_modules_se_layer_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_104 = l_self_modules_backbone_modules_layer2_modules_5_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_5_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_37 = torch.nn.functional.relu(x_36, inplace=True)
        x_36 = None
        x_38 = torch.conv2d(
            x_37,
            l_self_modules_backbone_modules_layer2_modules_5_modules_se_layer_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_5_modules_se_layer_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_37 = l_self_modules_backbone_modules_layer2_modules_5_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_5_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = (None)
        x_39 = torch.sigmoid(x_38)
        x_38 = None
        out_105 = out_103 * x_39
        out_103 = x_39 = None
        out_105 += out_95
        out_106 = out_105
        out_105 = out_95 = None
        out_107 = torch.nn.functional.relu(out_106, inplace=True)
        out_106 = None
        out_108 = torch.conv2d(
            out_107,
            l_self_modules_backbone_modules_layer2_modules_6_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer2_modules_6_modules_conv1_parameters_weight_ = (
            None
        )
        out_109 = torch.nn.functional.batch_norm(
            out_108,
            l_self_modules_backbone_modules_layer2_modules_6_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_6_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_6_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_6_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_108 = l_self_modules_backbone_modules_layer2_modules_6_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_6_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_6_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_6_modules_bn1_parameters_bias_ = (None)
        out_110 = torch.nn.functional.relu(out_109, inplace=True)
        out_109 = None
        out_111 = torch.conv2d(
            out_110,
            l_self_modules_backbone_modules_layer2_modules_6_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_110 = l_self_modules_backbone_modules_layer2_modules_6_modules_conv2_parameters_weight_ = (None)
        out_112 = torch.nn.functional.batch_norm(
            out_111,
            l_self_modules_backbone_modules_layer2_modules_6_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_6_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_6_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_6_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_111 = l_self_modules_backbone_modules_layer2_modules_6_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_6_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_6_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_6_modules_bn2_parameters_bias_ = (None)
        out_113 = torch.nn.functional.relu(out_112, inplace=True)
        out_112 = None
        out_114 = torch.conv2d(
            out_113,
            l_self_modules_backbone_modules_layer2_modules_6_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_113 = l_self_modules_backbone_modules_layer2_modules_6_modules_conv3_parameters_weight_ = (None)
        out_115 = torch.nn.functional.batch_norm(
            out_114,
            l_self_modules_backbone_modules_layer2_modules_6_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_6_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_6_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_6_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_114 = l_self_modules_backbone_modules_layer2_modules_6_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_6_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_6_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_6_modules_bn3_parameters_bias_ = (None)
        out_116 = torch.nn.functional.adaptive_avg_pool2d(out_115, 1)
        x_40 = torch.conv2d(
            out_116,
            l_self_modules_backbone_modules_layer2_modules_6_modules_se_layer_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_6_modules_se_layer_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_116 = l_self_modules_backbone_modules_layer2_modules_6_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_6_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_41 = torch.nn.functional.relu(x_40, inplace=True)
        x_40 = None
        x_42 = torch.conv2d(
            x_41,
            l_self_modules_backbone_modules_layer2_modules_6_modules_se_layer_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_6_modules_se_layer_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_41 = l_self_modules_backbone_modules_layer2_modules_6_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_6_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = (None)
        x_43 = torch.sigmoid(x_42)
        x_42 = None
        out_117 = out_115 * x_43
        out_115 = x_43 = None
        out_117 += out_107
        out_118 = out_117
        out_117 = out_107 = None
        out_119 = torch.nn.functional.relu(out_118, inplace=True)
        out_118 = None
        out_120 = torch.conv2d(
            out_119,
            l_self_modules_backbone_modules_layer2_modules_7_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer2_modules_7_modules_conv1_parameters_weight_ = (
            None
        )
        out_121 = torch.nn.functional.batch_norm(
            out_120,
            l_self_modules_backbone_modules_layer2_modules_7_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_7_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_7_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_7_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_120 = l_self_modules_backbone_modules_layer2_modules_7_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_7_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_7_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_7_modules_bn1_parameters_bias_ = (None)
        out_122 = torch.nn.functional.relu(out_121, inplace=True)
        out_121 = None
        out_123 = torch.conv2d(
            out_122,
            l_self_modules_backbone_modules_layer2_modules_7_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_122 = l_self_modules_backbone_modules_layer2_modules_7_modules_conv2_parameters_weight_ = (None)
        out_124 = torch.nn.functional.batch_norm(
            out_123,
            l_self_modules_backbone_modules_layer2_modules_7_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_7_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_7_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_7_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_123 = l_self_modules_backbone_modules_layer2_modules_7_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_7_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_7_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_7_modules_bn2_parameters_bias_ = (None)
        out_125 = torch.nn.functional.relu(out_124, inplace=True)
        out_124 = None
        out_126 = torch.conv2d(
            out_125,
            l_self_modules_backbone_modules_layer2_modules_7_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_125 = l_self_modules_backbone_modules_layer2_modules_7_modules_conv3_parameters_weight_ = (None)
        out_127 = torch.nn.functional.batch_norm(
            out_126,
            l_self_modules_backbone_modules_layer2_modules_7_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_7_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_7_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_7_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_126 = l_self_modules_backbone_modules_layer2_modules_7_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_7_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_7_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_7_modules_bn3_parameters_bias_ = (None)
        out_128 = torch.nn.functional.adaptive_avg_pool2d(out_127, 1)
        x_44 = torch.conv2d(
            out_128,
            l_self_modules_backbone_modules_layer2_modules_7_modules_se_layer_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_7_modules_se_layer_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_128 = l_self_modules_backbone_modules_layer2_modules_7_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_7_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_45 = torch.nn.functional.relu(x_44, inplace=True)
        x_44 = None
        x_46 = torch.conv2d(
            x_45,
            l_self_modules_backbone_modules_layer2_modules_7_modules_se_layer_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_7_modules_se_layer_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_45 = l_self_modules_backbone_modules_layer2_modules_7_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_7_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = (None)
        x_47 = torch.sigmoid(x_46)
        x_46 = None
        out_129 = out_127 * x_47
        out_127 = x_47 = None
        out_129 += out_119
        out_130 = out_129
        out_129 = out_119 = None
        out_131 = torch.nn.functional.relu(out_130, inplace=True)
        out_130 = None
        out_132 = torch.conv2d(
            out_131,
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
        out_133 = torch.nn.functional.batch_norm(
            out_132,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_132 = l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_bias_ = (None)
        out_134 = torch.nn.functional.relu(out_133, inplace=True)
        out_133 = None
        out_135 = torch.conv2d(
            out_134,
            l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        out_134 = l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_parameters_weight_ = (None)
        out_136 = torch.nn.functional.batch_norm(
            out_135,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_135 = l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_bias_ = (None)
        out_137 = torch.nn.functional.relu(out_136, inplace=True)
        out_136 = None
        out_138 = torch.conv2d(
            out_137,
            l_self_modules_backbone_modules_layer3_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_137 = l_self_modules_backbone_modules_layer3_modules_0_modules_conv3_parameters_weight_ = (None)
        out_139 = torch.nn.functional.batch_norm(
            out_138,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_138 = l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_bias_ = (None)
        out_140 = torch.nn.functional.adaptive_avg_pool2d(out_139, 1)
        x_48 = torch.conv2d(
            out_140,
            l_self_modules_backbone_modules_layer3_modules_0_modules_se_layer_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_se_layer_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_140 = l_self_modules_backbone_modules_layer3_modules_0_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_49 = torch.nn.functional.relu(x_48, inplace=True)
        x_48 = None
        x_50 = torch.conv2d(
            x_49,
            l_self_modules_backbone_modules_layer3_modules_0_modules_se_layer_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_se_layer_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_49 = l_self_modules_backbone_modules_layer3_modules_0_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = (None)
        x_51 = torch.sigmoid(x_50)
        x_50 = None
        out_141 = out_139 * x_51
        out_139 = x_51 = None
        input_5 = torch.conv2d(
            out_131,
            l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        out_131 = l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_6 = torch.nn.functional.batch_norm(
            input_5,
            l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_5 = l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        out_141 += input_6
        out_142 = out_141
        out_141 = input_6 = None
        out_143 = torch.nn.functional.relu(out_142, inplace=True)
        out_142 = None
        out_144 = torch.conv2d(
            out_143,
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
        out_145 = torch.nn.functional.batch_norm(
            out_144,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_144 = l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_bias_ = (None)
        out_146 = torch.nn.functional.relu(out_145, inplace=True)
        out_145 = None
        out_147 = torch.conv2d(
            out_146,
            l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_146 = l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_parameters_weight_ = (None)
        out_148 = torch.nn.functional.batch_norm(
            out_147,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_147 = l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_bias_ = (None)
        out_149 = torch.nn.functional.relu(out_148, inplace=True)
        out_148 = None
        out_150 = torch.conv2d(
            out_149,
            l_self_modules_backbone_modules_layer3_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_149 = l_self_modules_backbone_modules_layer3_modules_1_modules_conv3_parameters_weight_ = (None)
        out_151 = torch.nn.functional.batch_norm(
            out_150,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_150 = l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_parameters_bias_ = (None)
        out_152 = torch.nn.functional.adaptive_avg_pool2d(out_151, 1)
        x_52 = torch.conv2d(
            out_152,
            l_self_modules_backbone_modules_layer3_modules_1_modules_se_layer_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_se_layer_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_152 = l_self_modules_backbone_modules_layer3_modules_1_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_1_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_53 = torch.nn.functional.relu(x_52, inplace=True)
        x_52 = None
        x_54 = torch.conv2d(
            x_53,
            l_self_modules_backbone_modules_layer3_modules_1_modules_se_layer_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_se_layer_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_53 = l_self_modules_backbone_modules_layer3_modules_1_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_1_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = (None)
        x_55 = torch.sigmoid(x_54)
        x_54 = None
        out_153 = out_151 * x_55
        out_151 = x_55 = None
        out_153 += out_143
        out_154 = out_153
        out_153 = out_143 = None
        out_155 = torch.nn.functional.relu(out_154, inplace=True)
        out_154 = None
        out_156 = torch.conv2d(
            out_155,
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
        out_157 = torch.nn.functional.batch_norm(
            out_156,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_156 = l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_parameters_bias_ = (None)
        out_158 = torch.nn.functional.relu(out_157, inplace=True)
        out_157 = None
        out_159 = torch.conv2d(
            out_158,
            l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_158 = l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_parameters_weight_ = (None)
        out_160 = torch.nn.functional.batch_norm(
            out_159,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_159 = l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_parameters_bias_ = (None)
        out_161 = torch.nn.functional.relu(out_160, inplace=True)
        out_160 = None
        out_162 = torch.conv2d(
            out_161,
            l_self_modules_backbone_modules_layer3_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_161 = l_self_modules_backbone_modules_layer3_modules_2_modules_conv3_parameters_weight_ = (None)
        out_163 = torch.nn.functional.batch_norm(
            out_162,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_162 = l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_parameters_bias_ = (None)
        out_164 = torch.nn.functional.adaptive_avg_pool2d(out_163, 1)
        x_56 = torch.conv2d(
            out_164,
            l_self_modules_backbone_modules_layer3_modules_2_modules_se_layer_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_se_layer_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_164 = l_self_modules_backbone_modules_layer3_modules_2_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_2_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_57 = torch.nn.functional.relu(x_56, inplace=True)
        x_56 = None
        x_58 = torch.conv2d(
            x_57,
            l_self_modules_backbone_modules_layer3_modules_2_modules_se_layer_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_se_layer_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_57 = l_self_modules_backbone_modules_layer3_modules_2_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_2_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = (None)
        x_59 = torch.sigmoid(x_58)
        x_58 = None
        out_165 = out_163 * x_59
        out_163 = x_59 = None
        out_165 += out_155
        out_166 = out_165
        out_165 = out_155 = None
        out_167 = torch.nn.functional.relu(out_166, inplace=True)
        out_166 = None
        out_168 = torch.conv2d(
            out_167,
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
        out_169 = torch.nn.functional.batch_norm(
            out_168,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_168 = l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_parameters_bias_ = (None)
        out_170 = torch.nn.functional.relu(out_169, inplace=True)
        out_169 = None
        out_171 = torch.conv2d(
            out_170,
            l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_170 = l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_parameters_weight_ = (None)
        out_172 = torch.nn.functional.batch_norm(
            out_171,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_171 = l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_parameters_bias_ = (None)
        out_173 = torch.nn.functional.relu(out_172, inplace=True)
        out_172 = None
        out_174 = torch.conv2d(
            out_173,
            l_self_modules_backbone_modules_layer3_modules_3_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_173 = l_self_modules_backbone_modules_layer3_modules_3_modules_conv3_parameters_weight_ = (None)
        out_175 = torch.nn.functional.batch_norm(
            out_174,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_174 = l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_parameters_bias_ = (None)
        out_176 = torch.nn.functional.adaptive_avg_pool2d(out_175, 1)
        x_60 = torch.conv2d(
            out_176,
            l_self_modules_backbone_modules_layer3_modules_3_modules_se_layer_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_se_layer_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_176 = l_self_modules_backbone_modules_layer3_modules_3_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_3_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_61 = torch.nn.functional.relu(x_60, inplace=True)
        x_60 = None
        x_62 = torch.conv2d(
            x_61,
            l_self_modules_backbone_modules_layer3_modules_3_modules_se_layer_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_se_layer_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_61 = l_self_modules_backbone_modules_layer3_modules_3_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_3_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = (None)
        x_63 = torch.sigmoid(x_62)
        x_62 = None
        out_177 = out_175 * x_63
        out_175 = x_63 = None
        out_177 += out_167
        out_178 = out_177
        out_177 = out_167 = None
        out_179 = torch.nn.functional.relu(out_178, inplace=True)
        out_178 = None
        out_180 = torch.conv2d(
            out_179,
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
        out_181 = torch.nn.functional.batch_norm(
            out_180,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_180 = l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_parameters_bias_ = (None)
        out_182 = torch.nn.functional.relu(out_181, inplace=True)
        out_181 = None
        out_183 = torch.conv2d(
            out_182,
            l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_182 = l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_parameters_weight_ = (None)
        out_184 = torch.nn.functional.batch_norm(
            out_183,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_183 = l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_parameters_bias_ = (None)
        out_185 = torch.nn.functional.relu(out_184, inplace=True)
        out_184 = None
        out_186 = torch.conv2d(
            out_185,
            l_self_modules_backbone_modules_layer3_modules_4_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_185 = l_self_modules_backbone_modules_layer3_modules_4_modules_conv3_parameters_weight_ = (None)
        out_187 = torch.nn.functional.batch_norm(
            out_186,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_186 = l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_parameters_bias_ = (None)
        out_188 = torch.nn.functional.adaptive_avg_pool2d(out_187, 1)
        x_64 = torch.conv2d(
            out_188,
            l_self_modules_backbone_modules_layer3_modules_4_modules_se_layer_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_se_layer_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_188 = l_self_modules_backbone_modules_layer3_modules_4_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_4_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_65 = torch.nn.functional.relu(x_64, inplace=True)
        x_64 = None
        x_66 = torch.conv2d(
            x_65,
            l_self_modules_backbone_modules_layer3_modules_4_modules_se_layer_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_se_layer_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_65 = l_self_modules_backbone_modules_layer3_modules_4_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_4_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = (None)
        x_67 = torch.sigmoid(x_66)
        x_66 = None
        out_189 = out_187 * x_67
        out_187 = x_67 = None
        out_189 += out_179
        out_190 = out_189
        out_189 = out_179 = None
        out_191 = torch.nn.functional.relu(out_190, inplace=True)
        out_190 = None
        out_192 = torch.conv2d(
            out_191,
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
        out_193 = torch.nn.functional.batch_norm(
            out_192,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_192 = l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_parameters_bias_ = (None)
        out_194 = torch.nn.functional.relu(out_193, inplace=True)
        out_193 = None
        out_195 = torch.conv2d(
            out_194,
            l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_194 = l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_parameters_weight_ = (None)
        out_196 = torch.nn.functional.batch_norm(
            out_195,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_195 = l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_parameters_bias_ = (None)
        out_197 = torch.nn.functional.relu(out_196, inplace=True)
        out_196 = None
        out_198 = torch.conv2d(
            out_197,
            l_self_modules_backbone_modules_layer3_modules_5_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_197 = l_self_modules_backbone_modules_layer3_modules_5_modules_conv3_parameters_weight_ = (None)
        out_199 = torch.nn.functional.batch_norm(
            out_198,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_198 = l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_parameters_bias_ = (None)
        out_200 = torch.nn.functional.adaptive_avg_pool2d(out_199, 1)
        x_68 = torch.conv2d(
            out_200,
            l_self_modules_backbone_modules_layer3_modules_5_modules_se_layer_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_se_layer_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_200 = l_self_modules_backbone_modules_layer3_modules_5_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_5_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_69 = torch.nn.functional.relu(x_68, inplace=True)
        x_68 = None
        x_70 = torch.conv2d(
            x_69,
            l_self_modules_backbone_modules_layer3_modules_5_modules_se_layer_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_se_layer_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_69 = l_self_modules_backbone_modules_layer3_modules_5_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_5_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = (None)
        x_71 = torch.sigmoid(x_70)
        x_70 = None
        out_201 = out_199 * x_71
        out_199 = x_71 = None
        out_201 += out_191
        out_202 = out_201
        out_201 = out_191 = None
        out_203 = torch.nn.functional.relu(out_202, inplace=True)
        out_202 = None
        out_204 = torch.conv2d(
            out_203,
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
        out_205 = torch.nn.functional.batch_norm(
            out_204,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_204 = l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_parameters_bias_ = (None)
        out_206 = torch.nn.functional.relu(out_205, inplace=True)
        out_205 = None
        out_207 = torch.conv2d(
            out_206,
            l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_206 = l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_parameters_weight_ = (None)
        out_208 = torch.nn.functional.batch_norm(
            out_207,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_207 = l_self_modules_backbone_modules_layer3_modules_6_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_6_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_6_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_6_modules_bn2_parameters_bias_ = (None)
        out_209 = torch.nn.functional.relu(out_208, inplace=True)
        out_208 = None
        out_210 = torch.conv2d(
            out_209,
            l_self_modules_backbone_modules_layer3_modules_6_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_209 = l_self_modules_backbone_modules_layer3_modules_6_modules_conv3_parameters_weight_ = (None)
        out_211 = torch.nn.functional.batch_norm(
            out_210,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_210 = l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_parameters_bias_ = (None)
        out_212 = torch.nn.functional.adaptive_avg_pool2d(out_211, 1)
        x_72 = torch.conv2d(
            out_212,
            l_self_modules_backbone_modules_layer3_modules_6_modules_se_layer_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_se_layer_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_212 = l_self_modules_backbone_modules_layer3_modules_6_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_6_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_73 = torch.nn.functional.relu(x_72, inplace=True)
        x_72 = None
        x_74 = torch.conv2d(
            x_73,
            l_self_modules_backbone_modules_layer3_modules_6_modules_se_layer_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_se_layer_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_73 = l_self_modules_backbone_modules_layer3_modules_6_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_6_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = (None)
        x_75 = torch.sigmoid(x_74)
        x_74 = None
        out_213 = out_211 * x_75
        out_211 = x_75 = None
        out_213 += out_203
        out_214 = out_213
        out_213 = out_203 = None
        out_215 = torch.nn.functional.relu(out_214, inplace=True)
        out_214 = None
        out_216 = torch.conv2d(
            out_215,
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
        out_217 = torch.nn.functional.batch_norm(
            out_216,
            l_self_modules_backbone_modules_layer3_modules_7_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_216 = l_self_modules_backbone_modules_layer3_modules_7_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_7_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_7_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_7_modules_bn1_parameters_bias_ = (None)
        out_218 = torch.nn.functional.relu(out_217, inplace=True)
        out_217 = None
        out_219 = torch.conv2d(
            out_218,
            l_self_modules_backbone_modules_layer3_modules_7_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_218 = l_self_modules_backbone_modules_layer3_modules_7_modules_conv2_parameters_weight_ = (None)
        out_220 = torch.nn.functional.batch_norm(
            out_219,
            l_self_modules_backbone_modules_layer3_modules_7_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_219 = l_self_modules_backbone_modules_layer3_modules_7_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_7_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_7_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_7_modules_bn2_parameters_bias_ = (None)
        out_221 = torch.nn.functional.relu(out_220, inplace=True)
        out_220 = None
        out_222 = torch.conv2d(
            out_221,
            l_self_modules_backbone_modules_layer3_modules_7_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_221 = l_self_modules_backbone_modules_layer3_modules_7_modules_conv3_parameters_weight_ = (None)
        out_223 = torch.nn.functional.batch_norm(
            out_222,
            l_self_modules_backbone_modules_layer3_modules_7_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_222 = l_self_modules_backbone_modules_layer3_modules_7_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_7_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_7_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_7_modules_bn3_parameters_bias_ = (None)
        out_224 = torch.nn.functional.adaptive_avg_pool2d(out_223, 1)
        x_76 = torch.conv2d(
            out_224,
            l_self_modules_backbone_modules_layer3_modules_7_modules_se_layer_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_se_layer_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_224 = l_self_modules_backbone_modules_layer3_modules_7_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_7_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_77 = torch.nn.functional.relu(x_76, inplace=True)
        x_76 = None
        x_78 = torch.conv2d(
            x_77,
            l_self_modules_backbone_modules_layer3_modules_7_modules_se_layer_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_se_layer_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_77 = l_self_modules_backbone_modules_layer3_modules_7_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_7_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = (None)
        x_79 = torch.sigmoid(x_78)
        x_78 = None
        out_225 = out_223 * x_79
        out_223 = x_79 = None
        out_225 += out_215
        out_226 = out_225
        out_225 = out_215 = None
        out_227 = torch.nn.functional.relu(out_226, inplace=True)
        out_226 = None
        out_228 = torch.conv2d(
            out_227,
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
        out_229 = torch.nn.functional.batch_norm(
            out_228,
            l_self_modules_backbone_modules_layer3_modules_8_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_228 = l_self_modules_backbone_modules_layer3_modules_8_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_8_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_8_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_8_modules_bn1_parameters_bias_ = (None)
        out_230 = torch.nn.functional.relu(out_229, inplace=True)
        out_229 = None
        out_231 = torch.conv2d(
            out_230,
            l_self_modules_backbone_modules_layer3_modules_8_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_230 = l_self_modules_backbone_modules_layer3_modules_8_modules_conv2_parameters_weight_ = (None)
        out_232 = torch.nn.functional.batch_norm(
            out_231,
            l_self_modules_backbone_modules_layer3_modules_8_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_231 = l_self_modules_backbone_modules_layer3_modules_8_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_8_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_8_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_8_modules_bn2_parameters_bias_ = (None)
        out_233 = torch.nn.functional.relu(out_232, inplace=True)
        out_232 = None
        out_234 = torch.conv2d(
            out_233,
            l_self_modules_backbone_modules_layer3_modules_8_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_233 = l_self_modules_backbone_modules_layer3_modules_8_modules_conv3_parameters_weight_ = (None)
        out_235 = torch.nn.functional.batch_norm(
            out_234,
            l_self_modules_backbone_modules_layer3_modules_8_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_234 = l_self_modules_backbone_modules_layer3_modules_8_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_8_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_8_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_8_modules_bn3_parameters_bias_ = (None)
        out_236 = torch.nn.functional.adaptive_avg_pool2d(out_235, 1)
        x_80 = torch.conv2d(
            out_236,
            l_self_modules_backbone_modules_layer3_modules_8_modules_se_layer_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_se_layer_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_236 = l_self_modules_backbone_modules_layer3_modules_8_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_8_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_81 = torch.nn.functional.relu(x_80, inplace=True)
        x_80 = None
        x_82 = torch.conv2d(
            x_81,
            l_self_modules_backbone_modules_layer3_modules_8_modules_se_layer_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_se_layer_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_81 = l_self_modules_backbone_modules_layer3_modules_8_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_8_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = (None)
        x_83 = torch.sigmoid(x_82)
        x_82 = None
        out_237 = out_235 * x_83
        out_235 = x_83 = None
        out_237 += out_227
        out_238 = out_237
        out_237 = out_227 = None
        out_239 = torch.nn.functional.relu(out_238, inplace=True)
        out_238 = None
        out_240 = torch.conv2d(
            out_239,
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
        out_241 = torch.nn.functional.batch_norm(
            out_240,
            l_self_modules_backbone_modules_layer3_modules_9_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_240 = l_self_modules_backbone_modules_layer3_modules_9_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_9_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_9_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_9_modules_bn1_parameters_bias_ = (None)
        out_242 = torch.nn.functional.relu(out_241, inplace=True)
        out_241 = None
        out_243 = torch.conv2d(
            out_242,
            l_self_modules_backbone_modules_layer3_modules_9_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_242 = l_self_modules_backbone_modules_layer3_modules_9_modules_conv2_parameters_weight_ = (None)
        out_244 = torch.nn.functional.batch_norm(
            out_243,
            l_self_modules_backbone_modules_layer3_modules_9_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_243 = l_self_modules_backbone_modules_layer3_modules_9_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_9_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_9_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_9_modules_bn2_parameters_bias_ = (None)
        out_245 = torch.nn.functional.relu(out_244, inplace=True)
        out_244 = None
        out_246 = torch.conv2d(
            out_245,
            l_self_modules_backbone_modules_layer3_modules_9_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_245 = l_self_modules_backbone_modules_layer3_modules_9_modules_conv3_parameters_weight_ = (None)
        out_247 = torch.nn.functional.batch_norm(
            out_246,
            l_self_modules_backbone_modules_layer3_modules_9_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_246 = l_self_modules_backbone_modules_layer3_modules_9_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_9_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_9_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_9_modules_bn3_parameters_bias_ = (None)
        out_248 = torch.nn.functional.adaptive_avg_pool2d(out_247, 1)
        x_84 = torch.conv2d(
            out_248,
            l_self_modules_backbone_modules_layer3_modules_9_modules_se_layer_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_se_layer_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_248 = l_self_modules_backbone_modules_layer3_modules_9_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_9_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_85 = torch.nn.functional.relu(x_84, inplace=True)
        x_84 = None
        x_86 = torch.conv2d(
            x_85,
            l_self_modules_backbone_modules_layer3_modules_9_modules_se_layer_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_se_layer_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_85 = l_self_modules_backbone_modules_layer3_modules_9_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_9_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = (None)
        x_87 = torch.sigmoid(x_86)
        x_86 = None
        out_249 = out_247 * x_87
        out_247 = x_87 = None
        out_249 += out_239
        out_250 = out_249
        out_249 = out_239 = None
        out_251 = torch.nn.functional.relu(out_250, inplace=True)
        out_250 = None
        out_252 = torch.conv2d(
            out_251,
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
        out_253 = torch.nn.functional.batch_norm(
            out_252,
            l_self_modules_backbone_modules_layer3_modules_10_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_252 = l_self_modules_backbone_modules_layer3_modules_10_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_10_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_10_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_10_modules_bn1_parameters_bias_ = (None)
        out_254 = torch.nn.functional.relu(out_253, inplace=True)
        out_253 = None
        out_255 = torch.conv2d(
            out_254,
            l_self_modules_backbone_modules_layer3_modules_10_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_254 = l_self_modules_backbone_modules_layer3_modules_10_modules_conv2_parameters_weight_ = (None)
        out_256 = torch.nn.functional.batch_norm(
            out_255,
            l_self_modules_backbone_modules_layer3_modules_10_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_255 = l_self_modules_backbone_modules_layer3_modules_10_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_10_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_10_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_10_modules_bn2_parameters_bias_ = (None)
        out_257 = torch.nn.functional.relu(out_256, inplace=True)
        out_256 = None
        out_258 = torch.conv2d(
            out_257,
            l_self_modules_backbone_modules_layer3_modules_10_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_257 = l_self_modules_backbone_modules_layer3_modules_10_modules_conv3_parameters_weight_ = (None)
        out_259 = torch.nn.functional.batch_norm(
            out_258,
            l_self_modules_backbone_modules_layer3_modules_10_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_258 = l_self_modules_backbone_modules_layer3_modules_10_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_10_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_10_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_10_modules_bn3_parameters_bias_ = (None)
        out_260 = torch.nn.functional.adaptive_avg_pool2d(out_259, 1)
        x_88 = torch.conv2d(
            out_260,
            l_self_modules_backbone_modules_layer3_modules_10_modules_se_layer_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_se_layer_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_260 = l_self_modules_backbone_modules_layer3_modules_10_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_10_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_89 = torch.nn.functional.relu(x_88, inplace=True)
        x_88 = None
        x_90 = torch.conv2d(
            x_89,
            l_self_modules_backbone_modules_layer3_modules_10_modules_se_layer_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_se_layer_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_89 = l_self_modules_backbone_modules_layer3_modules_10_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_10_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = (None)
        x_91 = torch.sigmoid(x_90)
        x_90 = None
        out_261 = out_259 * x_91
        out_259 = x_91 = None
        out_261 += out_251
        out_262 = out_261
        out_261 = out_251 = None
        out_263 = torch.nn.functional.relu(out_262, inplace=True)
        out_262 = None
        out_264 = torch.conv2d(
            out_263,
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
        out_265 = torch.nn.functional.batch_norm(
            out_264,
            l_self_modules_backbone_modules_layer3_modules_11_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_264 = l_self_modules_backbone_modules_layer3_modules_11_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_11_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_11_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_11_modules_bn1_parameters_bias_ = (None)
        out_266 = torch.nn.functional.relu(out_265, inplace=True)
        out_265 = None
        out_267 = torch.conv2d(
            out_266,
            l_self_modules_backbone_modules_layer3_modules_11_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_266 = l_self_modules_backbone_modules_layer3_modules_11_modules_conv2_parameters_weight_ = (None)
        out_268 = torch.nn.functional.batch_norm(
            out_267,
            l_self_modules_backbone_modules_layer3_modules_11_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_267 = l_self_modules_backbone_modules_layer3_modules_11_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_11_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_11_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_11_modules_bn2_parameters_bias_ = (None)
        out_269 = torch.nn.functional.relu(out_268, inplace=True)
        out_268 = None
        out_270 = torch.conv2d(
            out_269,
            l_self_modules_backbone_modules_layer3_modules_11_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_269 = l_self_modules_backbone_modules_layer3_modules_11_modules_conv3_parameters_weight_ = (None)
        out_271 = torch.nn.functional.batch_norm(
            out_270,
            l_self_modules_backbone_modules_layer3_modules_11_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_270 = l_self_modules_backbone_modules_layer3_modules_11_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_11_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_11_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_11_modules_bn3_parameters_bias_ = (None)
        out_272 = torch.nn.functional.adaptive_avg_pool2d(out_271, 1)
        x_92 = torch.conv2d(
            out_272,
            l_self_modules_backbone_modules_layer3_modules_11_modules_se_layer_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_se_layer_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_272 = l_self_modules_backbone_modules_layer3_modules_11_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_11_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_93 = torch.nn.functional.relu(x_92, inplace=True)
        x_92 = None
        x_94 = torch.conv2d(
            x_93,
            l_self_modules_backbone_modules_layer3_modules_11_modules_se_layer_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_se_layer_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_93 = l_self_modules_backbone_modules_layer3_modules_11_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_11_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = (None)
        x_95 = torch.sigmoid(x_94)
        x_94 = None
        out_273 = out_271 * x_95
        out_271 = x_95 = None
        out_273 += out_263
        out_274 = out_273
        out_273 = out_263 = None
        out_275 = torch.nn.functional.relu(out_274, inplace=True)
        out_274 = None
        out_276 = torch.conv2d(
            out_275,
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
        out_277 = torch.nn.functional.batch_norm(
            out_276,
            l_self_modules_backbone_modules_layer3_modules_12_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_276 = l_self_modules_backbone_modules_layer3_modules_12_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_12_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_12_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_12_modules_bn1_parameters_bias_ = (None)
        out_278 = torch.nn.functional.relu(out_277, inplace=True)
        out_277 = None
        out_279 = torch.conv2d(
            out_278,
            l_self_modules_backbone_modules_layer3_modules_12_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_278 = l_self_modules_backbone_modules_layer3_modules_12_modules_conv2_parameters_weight_ = (None)
        out_280 = torch.nn.functional.batch_norm(
            out_279,
            l_self_modules_backbone_modules_layer3_modules_12_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_279 = l_self_modules_backbone_modules_layer3_modules_12_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_12_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_12_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_12_modules_bn2_parameters_bias_ = (None)
        out_281 = torch.nn.functional.relu(out_280, inplace=True)
        out_280 = None
        out_282 = torch.conv2d(
            out_281,
            l_self_modules_backbone_modules_layer3_modules_12_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_281 = l_self_modules_backbone_modules_layer3_modules_12_modules_conv3_parameters_weight_ = (None)
        out_283 = torch.nn.functional.batch_norm(
            out_282,
            l_self_modules_backbone_modules_layer3_modules_12_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_282 = l_self_modules_backbone_modules_layer3_modules_12_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_12_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_12_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_12_modules_bn3_parameters_bias_ = (None)
        out_284 = torch.nn.functional.adaptive_avg_pool2d(out_283, 1)
        x_96 = torch.conv2d(
            out_284,
            l_self_modules_backbone_modules_layer3_modules_12_modules_se_layer_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_se_layer_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_284 = l_self_modules_backbone_modules_layer3_modules_12_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_12_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_97 = torch.nn.functional.relu(x_96, inplace=True)
        x_96 = None
        x_98 = torch.conv2d(
            x_97,
            l_self_modules_backbone_modules_layer3_modules_12_modules_se_layer_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_se_layer_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_97 = l_self_modules_backbone_modules_layer3_modules_12_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_12_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = (None)
        x_99 = torch.sigmoid(x_98)
        x_98 = None
        out_285 = out_283 * x_99
        out_283 = x_99 = None
        out_285 += out_275
        out_286 = out_285
        out_285 = out_275 = None
        out_287 = torch.nn.functional.relu(out_286, inplace=True)
        out_286 = None
        out_288 = torch.conv2d(
            out_287,
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
        out_289 = torch.nn.functional.batch_norm(
            out_288,
            l_self_modules_backbone_modules_layer3_modules_13_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_288 = l_self_modules_backbone_modules_layer3_modules_13_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_13_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_13_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_13_modules_bn1_parameters_bias_ = (None)
        out_290 = torch.nn.functional.relu(out_289, inplace=True)
        out_289 = None
        out_291 = torch.conv2d(
            out_290,
            l_self_modules_backbone_modules_layer3_modules_13_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_290 = l_self_modules_backbone_modules_layer3_modules_13_modules_conv2_parameters_weight_ = (None)
        out_292 = torch.nn.functional.batch_norm(
            out_291,
            l_self_modules_backbone_modules_layer3_modules_13_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_291 = l_self_modules_backbone_modules_layer3_modules_13_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_13_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_13_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_13_modules_bn2_parameters_bias_ = (None)
        out_293 = torch.nn.functional.relu(out_292, inplace=True)
        out_292 = None
        out_294 = torch.conv2d(
            out_293,
            l_self_modules_backbone_modules_layer3_modules_13_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_293 = l_self_modules_backbone_modules_layer3_modules_13_modules_conv3_parameters_weight_ = (None)
        out_295 = torch.nn.functional.batch_norm(
            out_294,
            l_self_modules_backbone_modules_layer3_modules_13_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_294 = l_self_modules_backbone_modules_layer3_modules_13_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_13_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_13_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_13_modules_bn3_parameters_bias_ = (None)
        out_296 = torch.nn.functional.adaptive_avg_pool2d(out_295, 1)
        x_100 = torch.conv2d(
            out_296,
            l_self_modules_backbone_modules_layer3_modules_13_modules_se_layer_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_se_layer_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_296 = l_self_modules_backbone_modules_layer3_modules_13_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_13_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_101 = torch.nn.functional.relu(x_100, inplace=True)
        x_100 = None
        x_102 = torch.conv2d(
            x_101,
            l_self_modules_backbone_modules_layer3_modules_13_modules_se_layer_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_se_layer_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_101 = l_self_modules_backbone_modules_layer3_modules_13_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_13_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = (None)
        x_103 = torch.sigmoid(x_102)
        x_102 = None
        out_297 = out_295 * x_103
        out_295 = x_103 = None
        out_297 += out_287
        out_298 = out_297
        out_297 = out_287 = None
        out_299 = torch.nn.functional.relu(out_298, inplace=True)
        out_298 = None
        out_300 = torch.conv2d(
            out_299,
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
        out_301 = torch.nn.functional.batch_norm(
            out_300,
            l_self_modules_backbone_modules_layer3_modules_14_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_300 = l_self_modules_backbone_modules_layer3_modules_14_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_14_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_14_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_14_modules_bn1_parameters_bias_ = (None)
        out_302 = torch.nn.functional.relu(out_301, inplace=True)
        out_301 = None
        out_303 = torch.conv2d(
            out_302,
            l_self_modules_backbone_modules_layer3_modules_14_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_302 = l_self_modules_backbone_modules_layer3_modules_14_modules_conv2_parameters_weight_ = (None)
        out_304 = torch.nn.functional.batch_norm(
            out_303,
            l_self_modules_backbone_modules_layer3_modules_14_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_303 = l_self_modules_backbone_modules_layer3_modules_14_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_14_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_14_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_14_modules_bn2_parameters_bias_ = (None)
        out_305 = torch.nn.functional.relu(out_304, inplace=True)
        out_304 = None
        out_306 = torch.conv2d(
            out_305,
            l_self_modules_backbone_modules_layer3_modules_14_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_305 = l_self_modules_backbone_modules_layer3_modules_14_modules_conv3_parameters_weight_ = (None)
        out_307 = torch.nn.functional.batch_norm(
            out_306,
            l_self_modules_backbone_modules_layer3_modules_14_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_306 = l_self_modules_backbone_modules_layer3_modules_14_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_14_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_14_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_14_modules_bn3_parameters_bias_ = (None)
        out_308 = torch.nn.functional.adaptive_avg_pool2d(out_307, 1)
        x_104 = torch.conv2d(
            out_308,
            l_self_modules_backbone_modules_layer3_modules_14_modules_se_layer_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_se_layer_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_308 = l_self_modules_backbone_modules_layer3_modules_14_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_14_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_105 = torch.nn.functional.relu(x_104, inplace=True)
        x_104 = None
        x_106 = torch.conv2d(
            x_105,
            l_self_modules_backbone_modules_layer3_modules_14_modules_se_layer_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_se_layer_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_105 = l_self_modules_backbone_modules_layer3_modules_14_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_14_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = (None)
        x_107 = torch.sigmoid(x_106)
        x_106 = None
        out_309 = out_307 * x_107
        out_307 = x_107 = None
        out_309 += out_299
        out_310 = out_309
        out_309 = out_299 = None
        out_311 = torch.nn.functional.relu(out_310, inplace=True)
        out_310 = None
        out_312 = torch.conv2d(
            out_311,
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
        out_313 = torch.nn.functional.batch_norm(
            out_312,
            l_self_modules_backbone_modules_layer3_modules_15_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_312 = l_self_modules_backbone_modules_layer3_modules_15_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_15_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_15_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_15_modules_bn1_parameters_bias_ = (None)
        out_314 = torch.nn.functional.relu(out_313, inplace=True)
        out_313 = None
        out_315 = torch.conv2d(
            out_314,
            l_self_modules_backbone_modules_layer3_modules_15_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_314 = l_self_modules_backbone_modules_layer3_modules_15_modules_conv2_parameters_weight_ = (None)
        out_316 = torch.nn.functional.batch_norm(
            out_315,
            l_self_modules_backbone_modules_layer3_modules_15_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_315 = l_self_modules_backbone_modules_layer3_modules_15_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_15_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_15_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_15_modules_bn2_parameters_bias_ = (None)
        out_317 = torch.nn.functional.relu(out_316, inplace=True)
        out_316 = None
        out_318 = torch.conv2d(
            out_317,
            l_self_modules_backbone_modules_layer3_modules_15_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_317 = l_self_modules_backbone_modules_layer3_modules_15_modules_conv3_parameters_weight_ = (None)
        out_319 = torch.nn.functional.batch_norm(
            out_318,
            l_self_modules_backbone_modules_layer3_modules_15_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_318 = l_self_modules_backbone_modules_layer3_modules_15_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_15_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_15_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_15_modules_bn3_parameters_bias_ = (None)
        out_320 = torch.nn.functional.adaptive_avg_pool2d(out_319, 1)
        x_108 = torch.conv2d(
            out_320,
            l_self_modules_backbone_modules_layer3_modules_15_modules_se_layer_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_se_layer_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_320 = l_self_modules_backbone_modules_layer3_modules_15_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_15_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_109 = torch.nn.functional.relu(x_108, inplace=True)
        x_108 = None
        x_110 = torch.conv2d(
            x_109,
            l_self_modules_backbone_modules_layer3_modules_15_modules_se_layer_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_se_layer_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_109 = l_self_modules_backbone_modules_layer3_modules_15_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_15_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = (None)
        x_111 = torch.sigmoid(x_110)
        x_110 = None
        out_321 = out_319 * x_111
        out_319 = x_111 = None
        out_321 += out_311
        out_322 = out_321
        out_321 = out_311 = None
        out_323 = torch.nn.functional.relu(out_322, inplace=True)
        out_322 = None
        out_324 = torch.conv2d(
            out_323,
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
        out_325 = torch.nn.functional.batch_norm(
            out_324,
            l_self_modules_backbone_modules_layer3_modules_16_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_324 = l_self_modules_backbone_modules_layer3_modules_16_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_16_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_16_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_16_modules_bn1_parameters_bias_ = (None)
        out_326 = torch.nn.functional.relu(out_325, inplace=True)
        out_325 = None
        out_327 = torch.conv2d(
            out_326,
            l_self_modules_backbone_modules_layer3_modules_16_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_326 = l_self_modules_backbone_modules_layer3_modules_16_modules_conv2_parameters_weight_ = (None)
        out_328 = torch.nn.functional.batch_norm(
            out_327,
            l_self_modules_backbone_modules_layer3_modules_16_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_327 = l_self_modules_backbone_modules_layer3_modules_16_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_16_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_16_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_16_modules_bn2_parameters_bias_ = (None)
        out_329 = torch.nn.functional.relu(out_328, inplace=True)
        out_328 = None
        out_330 = torch.conv2d(
            out_329,
            l_self_modules_backbone_modules_layer3_modules_16_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_329 = l_self_modules_backbone_modules_layer3_modules_16_modules_conv3_parameters_weight_ = (None)
        out_331 = torch.nn.functional.batch_norm(
            out_330,
            l_self_modules_backbone_modules_layer3_modules_16_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_330 = l_self_modules_backbone_modules_layer3_modules_16_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_16_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_16_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_16_modules_bn3_parameters_bias_ = (None)
        out_332 = torch.nn.functional.adaptive_avg_pool2d(out_331, 1)
        x_112 = torch.conv2d(
            out_332,
            l_self_modules_backbone_modules_layer3_modules_16_modules_se_layer_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_se_layer_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_332 = l_self_modules_backbone_modules_layer3_modules_16_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_16_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_113 = torch.nn.functional.relu(x_112, inplace=True)
        x_112 = None
        x_114 = torch.conv2d(
            x_113,
            l_self_modules_backbone_modules_layer3_modules_16_modules_se_layer_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_se_layer_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_113 = l_self_modules_backbone_modules_layer3_modules_16_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_16_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = (None)
        x_115 = torch.sigmoid(x_114)
        x_114 = None
        out_333 = out_331 * x_115
        out_331 = x_115 = None
        out_333 += out_323
        out_334 = out_333
        out_333 = out_323 = None
        out_335 = torch.nn.functional.relu(out_334, inplace=True)
        out_334 = None
        out_336 = torch.conv2d(
            out_335,
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
        out_337 = torch.nn.functional.batch_norm(
            out_336,
            l_self_modules_backbone_modules_layer3_modules_17_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_336 = l_self_modules_backbone_modules_layer3_modules_17_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_17_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_17_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_17_modules_bn1_parameters_bias_ = (None)
        out_338 = torch.nn.functional.relu(out_337, inplace=True)
        out_337 = None
        out_339 = torch.conv2d(
            out_338,
            l_self_modules_backbone_modules_layer3_modules_17_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_338 = l_self_modules_backbone_modules_layer3_modules_17_modules_conv2_parameters_weight_ = (None)
        out_340 = torch.nn.functional.batch_norm(
            out_339,
            l_self_modules_backbone_modules_layer3_modules_17_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_339 = l_self_modules_backbone_modules_layer3_modules_17_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_17_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_17_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_17_modules_bn2_parameters_bias_ = (None)
        out_341 = torch.nn.functional.relu(out_340, inplace=True)
        out_340 = None
        out_342 = torch.conv2d(
            out_341,
            l_self_modules_backbone_modules_layer3_modules_17_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_341 = l_self_modules_backbone_modules_layer3_modules_17_modules_conv3_parameters_weight_ = (None)
        out_343 = torch.nn.functional.batch_norm(
            out_342,
            l_self_modules_backbone_modules_layer3_modules_17_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_342 = l_self_modules_backbone_modules_layer3_modules_17_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_17_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_17_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_17_modules_bn3_parameters_bias_ = (None)
        out_344 = torch.nn.functional.adaptive_avg_pool2d(out_343, 1)
        x_116 = torch.conv2d(
            out_344,
            l_self_modules_backbone_modules_layer3_modules_17_modules_se_layer_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_se_layer_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_344 = l_self_modules_backbone_modules_layer3_modules_17_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_17_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_117 = torch.nn.functional.relu(x_116, inplace=True)
        x_116 = None
        x_118 = torch.conv2d(
            x_117,
            l_self_modules_backbone_modules_layer3_modules_17_modules_se_layer_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_se_layer_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_117 = l_self_modules_backbone_modules_layer3_modules_17_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_17_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = (None)
        x_119 = torch.sigmoid(x_118)
        x_118 = None
        out_345 = out_343 * x_119
        out_343 = x_119 = None
        out_345 += out_335
        out_346 = out_345
        out_345 = out_335 = None
        out_347 = torch.nn.functional.relu(out_346, inplace=True)
        out_346 = None
        out_348 = torch.conv2d(
            out_347,
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
        out_349 = torch.nn.functional.batch_norm(
            out_348,
            l_self_modules_backbone_modules_layer3_modules_18_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_348 = l_self_modules_backbone_modules_layer3_modules_18_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_18_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_18_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_18_modules_bn1_parameters_bias_ = (None)
        out_350 = torch.nn.functional.relu(out_349, inplace=True)
        out_349 = None
        out_351 = torch.conv2d(
            out_350,
            l_self_modules_backbone_modules_layer3_modules_18_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_350 = l_self_modules_backbone_modules_layer3_modules_18_modules_conv2_parameters_weight_ = (None)
        out_352 = torch.nn.functional.batch_norm(
            out_351,
            l_self_modules_backbone_modules_layer3_modules_18_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_351 = l_self_modules_backbone_modules_layer3_modules_18_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_18_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_18_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_18_modules_bn2_parameters_bias_ = (None)
        out_353 = torch.nn.functional.relu(out_352, inplace=True)
        out_352 = None
        out_354 = torch.conv2d(
            out_353,
            l_self_modules_backbone_modules_layer3_modules_18_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_353 = l_self_modules_backbone_modules_layer3_modules_18_modules_conv3_parameters_weight_ = (None)
        out_355 = torch.nn.functional.batch_norm(
            out_354,
            l_self_modules_backbone_modules_layer3_modules_18_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_354 = l_self_modules_backbone_modules_layer3_modules_18_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_18_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_18_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_18_modules_bn3_parameters_bias_ = (None)
        out_356 = torch.nn.functional.adaptive_avg_pool2d(out_355, 1)
        x_120 = torch.conv2d(
            out_356,
            l_self_modules_backbone_modules_layer3_modules_18_modules_se_layer_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_se_layer_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_356 = l_self_modules_backbone_modules_layer3_modules_18_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_18_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_121 = torch.nn.functional.relu(x_120, inplace=True)
        x_120 = None
        x_122 = torch.conv2d(
            x_121,
            l_self_modules_backbone_modules_layer3_modules_18_modules_se_layer_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_se_layer_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_121 = l_self_modules_backbone_modules_layer3_modules_18_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_18_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = (None)
        x_123 = torch.sigmoid(x_122)
        x_122 = None
        out_357 = out_355 * x_123
        out_355 = x_123 = None
        out_357 += out_347
        out_358 = out_357
        out_357 = out_347 = None
        out_359 = torch.nn.functional.relu(out_358, inplace=True)
        out_358 = None
        out_360 = torch.conv2d(
            out_359,
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
        out_361 = torch.nn.functional.batch_norm(
            out_360,
            l_self_modules_backbone_modules_layer3_modules_19_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_360 = l_self_modules_backbone_modules_layer3_modules_19_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_19_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_19_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_19_modules_bn1_parameters_bias_ = (None)
        out_362 = torch.nn.functional.relu(out_361, inplace=True)
        out_361 = None
        out_363 = torch.conv2d(
            out_362,
            l_self_modules_backbone_modules_layer3_modules_19_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_362 = l_self_modules_backbone_modules_layer3_modules_19_modules_conv2_parameters_weight_ = (None)
        out_364 = torch.nn.functional.batch_norm(
            out_363,
            l_self_modules_backbone_modules_layer3_modules_19_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_363 = l_self_modules_backbone_modules_layer3_modules_19_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_19_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_19_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_19_modules_bn2_parameters_bias_ = (None)
        out_365 = torch.nn.functional.relu(out_364, inplace=True)
        out_364 = None
        out_366 = torch.conv2d(
            out_365,
            l_self_modules_backbone_modules_layer3_modules_19_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_365 = l_self_modules_backbone_modules_layer3_modules_19_modules_conv3_parameters_weight_ = (None)
        out_367 = torch.nn.functional.batch_norm(
            out_366,
            l_self_modules_backbone_modules_layer3_modules_19_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_366 = l_self_modules_backbone_modules_layer3_modules_19_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_19_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_19_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_19_modules_bn3_parameters_bias_ = (None)
        out_368 = torch.nn.functional.adaptive_avg_pool2d(out_367, 1)
        x_124 = torch.conv2d(
            out_368,
            l_self_modules_backbone_modules_layer3_modules_19_modules_se_layer_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_se_layer_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_368 = l_self_modules_backbone_modules_layer3_modules_19_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_19_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_125 = torch.nn.functional.relu(x_124, inplace=True)
        x_124 = None
        x_126 = torch.conv2d(
            x_125,
            l_self_modules_backbone_modules_layer3_modules_19_modules_se_layer_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_se_layer_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_125 = l_self_modules_backbone_modules_layer3_modules_19_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_19_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = (None)
        x_127 = torch.sigmoid(x_126)
        x_126 = None
        out_369 = out_367 * x_127
        out_367 = x_127 = None
        out_369 += out_359
        out_370 = out_369
        out_369 = out_359 = None
        out_371 = torch.nn.functional.relu(out_370, inplace=True)
        out_370 = None
        out_372 = torch.conv2d(
            out_371,
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
        out_373 = torch.nn.functional.batch_norm(
            out_372,
            l_self_modules_backbone_modules_layer3_modules_20_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_372 = l_self_modules_backbone_modules_layer3_modules_20_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_20_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_20_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_20_modules_bn1_parameters_bias_ = (None)
        out_374 = torch.nn.functional.relu(out_373, inplace=True)
        out_373 = None
        out_375 = torch.conv2d(
            out_374,
            l_self_modules_backbone_modules_layer3_modules_20_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_374 = l_self_modules_backbone_modules_layer3_modules_20_modules_conv2_parameters_weight_ = (None)
        out_376 = torch.nn.functional.batch_norm(
            out_375,
            l_self_modules_backbone_modules_layer3_modules_20_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_375 = l_self_modules_backbone_modules_layer3_modules_20_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_20_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_20_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_20_modules_bn2_parameters_bias_ = (None)
        out_377 = torch.nn.functional.relu(out_376, inplace=True)
        out_376 = None
        out_378 = torch.conv2d(
            out_377,
            l_self_modules_backbone_modules_layer3_modules_20_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_377 = l_self_modules_backbone_modules_layer3_modules_20_modules_conv3_parameters_weight_ = (None)
        out_379 = torch.nn.functional.batch_norm(
            out_378,
            l_self_modules_backbone_modules_layer3_modules_20_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_378 = l_self_modules_backbone_modules_layer3_modules_20_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_20_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_20_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_20_modules_bn3_parameters_bias_ = (None)
        out_380 = torch.nn.functional.adaptive_avg_pool2d(out_379, 1)
        x_128 = torch.conv2d(
            out_380,
            l_self_modules_backbone_modules_layer3_modules_20_modules_se_layer_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_se_layer_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_380 = l_self_modules_backbone_modules_layer3_modules_20_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_20_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_129 = torch.nn.functional.relu(x_128, inplace=True)
        x_128 = None
        x_130 = torch.conv2d(
            x_129,
            l_self_modules_backbone_modules_layer3_modules_20_modules_se_layer_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_se_layer_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_129 = l_self_modules_backbone_modules_layer3_modules_20_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_20_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = (None)
        x_131 = torch.sigmoid(x_130)
        x_130 = None
        out_381 = out_379 * x_131
        out_379 = x_131 = None
        out_381 += out_371
        out_382 = out_381
        out_381 = out_371 = None
        out_383 = torch.nn.functional.relu(out_382, inplace=True)
        out_382 = None
        out_384 = torch.conv2d(
            out_383,
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
        out_385 = torch.nn.functional.batch_norm(
            out_384,
            l_self_modules_backbone_modules_layer3_modules_21_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_384 = l_self_modules_backbone_modules_layer3_modules_21_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_21_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_21_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_21_modules_bn1_parameters_bias_ = (None)
        out_386 = torch.nn.functional.relu(out_385, inplace=True)
        out_385 = None
        out_387 = torch.conv2d(
            out_386,
            l_self_modules_backbone_modules_layer3_modules_21_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_386 = l_self_modules_backbone_modules_layer3_modules_21_modules_conv2_parameters_weight_ = (None)
        out_388 = torch.nn.functional.batch_norm(
            out_387,
            l_self_modules_backbone_modules_layer3_modules_21_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_387 = l_self_modules_backbone_modules_layer3_modules_21_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_21_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_21_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_21_modules_bn2_parameters_bias_ = (None)
        out_389 = torch.nn.functional.relu(out_388, inplace=True)
        out_388 = None
        out_390 = torch.conv2d(
            out_389,
            l_self_modules_backbone_modules_layer3_modules_21_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_389 = l_self_modules_backbone_modules_layer3_modules_21_modules_conv3_parameters_weight_ = (None)
        out_391 = torch.nn.functional.batch_norm(
            out_390,
            l_self_modules_backbone_modules_layer3_modules_21_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_390 = l_self_modules_backbone_modules_layer3_modules_21_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_21_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_21_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_21_modules_bn3_parameters_bias_ = (None)
        out_392 = torch.nn.functional.adaptive_avg_pool2d(out_391, 1)
        x_132 = torch.conv2d(
            out_392,
            l_self_modules_backbone_modules_layer3_modules_21_modules_se_layer_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_se_layer_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_392 = l_self_modules_backbone_modules_layer3_modules_21_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_21_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_133 = torch.nn.functional.relu(x_132, inplace=True)
        x_132 = None
        x_134 = torch.conv2d(
            x_133,
            l_self_modules_backbone_modules_layer3_modules_21_modules_se_layer_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_se_layer_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_133 = l_self_modules_backbone_modules_layer3_modules_21_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_21_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = (None)
        x_135 = torch.sigmoid(x_134)
        x_134 = None
        out_393 = out_391 * x_135
        out_391 = x_135 = None
        out_393 += out_383
        out_394 = out_393
        out_393 = out_383 = None
        out_395 = torch.nn.functional.relu(out_394, inplace=True)
        out_394 = None
        out_396 = torch.conv2d(
            out_395,
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
        out_397 = torch.nn.functional.batch_norm(
            out_396,
            l_self_modules_backbone_modules_layer3_modules_22_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_396 = l_self_modules_backbone_modules_layer3_modules_22_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_22_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_22_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_22_modules_bn1_parameters_bias_ = (None)
        out_398 = torch.nn.functional.relu(out_397, inplace=True)
        out_397 = None
        out_399 = torch.conv2d(
            out_398,
            l_self_modules_backbone_modules_layer3_modules_22_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_398 = l_self_modules_backbone_modules_layer3_modules_22_modules_conv2_parameters_weight_ = (None)
        out_400 = torch.nn.functional.batch_norm(
            out_399,
            l_self_modules_backbone_modules_layer3_modules_22_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_399 = l_self_modules_backbone_modules_layer3_modules_22_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_22_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_22_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_22_modules_bn2_parameters_bias_ = (None)
        out_401 = torch.nn.functional.relu(out_400, inplace=True)
        out_400 = None
        out_402 = torch.conv2d(
            out_401,
            l_self_modules_backbone_modules_layer3_modules_22_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_401 = l_self_modules_backbone_modules_layer3_modules_22_modules_conv3_parameters_weight_ = (None)
        out_403 = torch.nn.functional.batch_norm(
            out_402,
            l_self_modules_backbone_modules_layer3_modules_22_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_402 = l_self_modules_backbone_modules_layer3_modules_22_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_22_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_22_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_22_modules_bn3_parameters_bias_ = (None)
        out_404 = torch.nn.functional.adaptive_avg_pool2d(out_403, 1)
        x_136 = torch.conv2d(
            out_404,
            l_self_modules_backbone_modules_layer3_modules_22_modules_se_layer_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_se_layer_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_404 = l_self_modules_backbone_modules_layer3_modules_22_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_22_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_137 = torch.nn.functional.relu(x_136, inplace=True)
        x_136 = None
        x_138 = torch.conv2d(
            x_137,
            l_self_modules_backbone_modules_layer3_modules_22_modules_se_layer_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_se_layer_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_137 = l_self_modules_backbone_modules_layer3_modules_22_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_22_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = (None)
        x_139 = torch.sigmoid(x_138)
        x_138 = None
        out_405 = out_403 * x_139
        out_403 = x_139 = None
        out_405 += out_395
        out_406 = out_405
        out_405 = out_395 = None
        out_407 = torch.nn.functional.relu(out_406, inplace=True)
        out_406 = None
        out_408 = torch.conv2d(
            out_407,
            l_self_modules_backbone_modules_layer3_modules_23_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_23_modules_conv1_parameters_weight_ = (
            None
        )
        out_409 = torch.nn.functional.batch_norm(
            out_408,
            l_self_modules_backbone_modules_layer3_modules_23_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_23_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_23_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_23_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_408 = l_self_modules_backbone_modules_layer3_modules_23_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_23_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_23_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_23_modules_bn1_parameters_bias_ = (None)
        out_410 = torch.nn.functional.relu(out_409, inplace=True)
        out_409 = None
        out_411 = torch.conv2d(
            out_410,
            l_self_modules_backbone_modules_layer3_modules_23_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_410 = l_self_modules_backbone_modules_layer3_modules_23_modules_conv2_parameters_weight_ = (None)
        out_412 = torch.nn.functional.batch_norm(
            out_411,
            l_self_modules_backbone_modules_layer3_modules_23_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_23_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_23_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_23_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_411 = l_self_modules_backbone_modules_layer3_modules_23_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_23_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_23_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_23_modules_bn2_parameters_bias_ = (None)
        out_413 = torch.nn.functional.relu(out_412, inplace=True)
        out_412 = None
        out_414 = torch.conv2d(
            out_413,
            l_self_modules_backbone_modules_layer3_modules_23_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_413 = l_self_modules_backbone_modules_layer3_modules_23_modules_conv3_parameters_weight_ = (None)
        out_415 = torch.nn.functional.batch_norm(
            out_414,
            l_self_modules_backbone_modules_layer3_modules_23_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_23_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_23_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_23_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_414 = l_self_modules_backbone_modules_layer3_modules_23_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_23_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_23_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_23_modules_bn3_parameters_bias_ = (None)
        out_416 = torch.nn.functional.adaptive_avg_pool2d(out_415, 1)
        x_140 = torch.conv2d(
            out_416,
            l_self_modules_backbone_modules_layer3_modules_23_modules_se_layer_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_23_modules_se_layer_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_416 = l_self_modules_backbone_modules_layer3_modules_23_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_23_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_141 = torch.nn.functional.relu(x_140, inplace=True)
        x_140 = None
        x_142 = torch.conv2d(
            x_141,
            l_self_modules_backbone_modules_layer3_modules_23_modules_se_layer_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_23_modules_se_layer_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_141 = l_self_modules_backbone_modules_layer3_modules_23_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_23_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = (None)
        x_143 = torch.sigmoid(x_142)
        x_142 = None
        out_417 = out_415 * x_143
        out_415 = x_143 = None
        out_417 += out_407
        out_418 = out_417
        out_417 = out_407 = None
        out_419 = torch.nn.functional.relu(out_418, inplace=True)
        out_418 = None
        out_420 = torch.conv2d(
            out_419,
            l_self_modules_backbone_modules_layer3_modules_24_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_24_modules_conv1_parameters_weight_ = (
            None
        )
        out_421 = torch.nn.functional.batch_norm(
            out_420,
            l_self_modules_backbone_modules_layer3_modules_24_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_24_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_24_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_24_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_420 = l_self_modules_backbone_modules_layer3_modules_24_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_24_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_24_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_24_modules_bn1_parameters_bias_ = (None)
        out_422 = torch.nn.functional.relu(out_421, inplace=True)
        out_421 = None
        out_423 = torch.conv2d(
            out_422,
            l_self_modules_backbone_modules_layer3_modules_24_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_422 = l_self_modules_backbone_modules_layer3_modules_24_modules_conv2_parameters_weight_ = (None)
        out_424 = torch.nn.functional.batch_norm(
            out_423,
            l_self_modules_backbone_modules_layer3_modules_24_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_24_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_24_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_24_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_423 = l_self_modules_backbone_modules_layer3_modules_24_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_24_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_24_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_24_modules_bn2_parameters_bias_ = (None)
        out_425 = torch.nn.functional.relu(out_424, inplace=True)
        out_424 = None
        out_426 = torch.conv2d(
            out_425,
            l_self_modules_backbone_modules_layer3_modules_24_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_425 = l_self_modules_backbone_modules_layer3_modules_24_modules_conv3_parameters_weight_ = (None)
        out_427 = torch.nn.functional.batch_norm(
            out_426,
            l_self_modules_backbone_modules_layer3_modules_24_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_24_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_24_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_24_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_426 = l_self_modules_backbone_modules_layer3_modules_24_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_24_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_24_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_24_modules_bn3_parameters_bias_ = (None)
        out_428 = torch.nn.functional.adaptive_avg_pool2d(out_427, 1)
        x_144 = torch.conv2d(
            out_428,
            l_self_modules_backbone_modules_layer3_modules_24_modules_se_layer_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_24_modules_se_layer_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_428 = l_self_modules_backbone_modules_layer3_modules_24_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_24_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_145 = torch.nn.functional.relu(x_144, inplace=True)
        x_144 = None
        x_146 = torch.conv2d(
            x_145,
            l_self_modules_backbone_modules_layer3_modules_24_modules_se_layer_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_24_modules_se_layer_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_145 = l_self_modules_backbone_modules_layer3_modules_24_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_24_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = (None)
        x_147 = torch.sigmoid(x_146)
        x_146 = None
        out_429 = out_427 * x_147
        out_427 = x_147 = None
        out_429 += out_419
        out_430 = out_429
        out_429 = out_419 = None
        out_431 = torch.nn.functional.relu(out_430, inplace=True)
        out_430 = None
        out_432 = torch.conv2d(
            out_431,
            l_self_modules_backbone_modules_layer3_modules_25_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_25_modules_conv1_parameters_weight_ = (
            None
        )
        out_433 = torch.nn.functional.batch_norm(
            out_432,
            l_self_modules_backbone_modules_layer3_modules_25_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_25_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_25_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_25_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_432 = l_self_modules_backbone_modules_layer3_modules_25_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_25_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_25_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_25_modules_bn1_parameters_bias_ = (None)
        out_434 = torch.nn.functional.relu(out_433, inplace=True)
        out_433 = None
        out_435 = torch.conv2d(
            out_434,
            l_self_modules_backbone_modules_layer3_modules_25_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_434 = l_self_modules_backbone_modules_layer3_modules_25_modules_conv2_parameters_weight_ = (None)
        out_436 = torch.nn.functional.batch_norm(
            out_435,
            l_self_modules_backbone_modules_layer3_modules_25_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_25_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_25_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_25_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_435 = l_self_modules_backbone_modules_layer3_modules_25_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_25_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_25_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_25_modules_bn2_parameters_bias_ = (None)
        out_437 = torch.nn.functional.relu(out_436, inplace=True)
        out_436 = None
        out_438 = torch.conv2d(
            out_437,
            l_self_modules_backbone_modules_layer3_modules_25_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_437 = l_self_modules_backbone_modules_layer3_modules_25_modules_conv3_parameters_weight_ = (None)
        out_439 = torch.nn.functional.batch_norm(
            out_438,
            l_self_modules_backbone_modules_layer3_modules_25_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_25_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_25_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_25_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_438 = l_self_modules_backbone_modules_layer3_modules_25_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_25_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_25_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_25_modules_bn3_parameters_bias_ = (None)
        out_440 = torch.nn.functional.adaptive_avg_pool2d(out_439, 1)
        x_148 = torch.conv2d(
            out_440,
            l_self_modules_backbone_modules_layer3_modules_25_modules_se_layer_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_25_modules_se_layer_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_440 = l_self_modules_backbone_modules_layer3_modules_25_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_25_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_149 = torch.nn.functional.relu(x_148, inplace=True)
        x_148 = None
        x_150 = torch.conv2d(
            x_149,
            l_self_modules_backbone_modules_layer3_modules_25_modules_se_layer_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_25_modules_se_layer_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_149 = l_self_modules_backbone_modules_layer3_modules_25_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_25_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = (None)
        x_151 = torch.sigmoid(x_150)
        x_150 = None
        out_441 = out_439 * x_151
        out_439 = x_151 = None
        out_441 += out_431
        out_442 = out_441
        out_441 = out_431 = None
        out_443 = torch.nn.functional.relu(out_442, inplace=True)
        out_442 = None
        out_444 = torch.conv2d(
            out_443,
            l_self_modules_backbone_modules_layer3_modules_26_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_26_modules_conv1_parameters_weight_ = (
            None
        )
        out_445 = torch.nn.functional.batch_norm(
            out_444,
            l_self_modules_backbone_modules_layer3_modules_26_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_26_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_26_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_26_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_444 = l_self_modules_backbone_modules_layer3_modules_26_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_26_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_26_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_26_modules_bn1_parameters_bias_ = (None)
        out_446 = torch.nn.functional.relu(out_445, inplace=True)
        out_445 = None
        out_447 = torch.conv2d(
            out_446,
            l_self_modules_backbone_modules_layer3_modules_26_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_446 = l_self_modules_backbone_modules_layer3_modules_26_modules_conv2_parameters_weight_ = (None)
        out_448 = torch.nn.functional.batch_norm(
            out_447,
            l_self_modules_backbone_modules_layer3_modules_26_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_26_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_26_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_26_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_447 = l_self_modules_backbone_modules_layer3_modules_26_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_26_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_26_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_26_modules_bn2_parameters_bias_ = (None)
        out_449 = torch.nn.functional.relu(out_448, inplace=True)
        out_448 = None
        out_450 = torch.conv2d(
            out_449,
            l_self_modules_backbone_modules_layer3_modules_26_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_449 = l_self_modules_backbone_modules_layer3_modules_26_modules_conv3_parameters_weight_ = (None)
        out_451 = torch.nn.functional.batch_norm(
            out_450,
            l_self_modules_backbone_modules_layer3_modules_26_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_26_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_26_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_26_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_450 = l_self_modules_backbone_modules_layer3_modules_26_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_26_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_26_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_26_modules_bn3_parameters_bias_ = (None)
        out_452 = torch.nn.functional.adaptive_avg_pool2d(out_451, 1)
        x_152 = torch.conv2d(
            out_452,
            l_self_modules_backbone_modules_layer3_modules_26_modules_se_layer_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_26_modules_se_layer_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_452 = l_self_modules_backbone_modules_layer3_modules_26_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_26_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_153 = torch.nn.functional.relu(x_152, inplace=True)
        x_152 = None
        x_154 = torch.conv2d(
            x_153,
            l_self_modules_backbone_modules_layer3_modules_26_modules_se_layer_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_26_modules_se_layer_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_153 = l_self_modules_backbone_modules_layer3_modules_26_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_26_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = (None)
        x_155 = torch.sigmoid(x_154)
        x_154 = None
        out_453 = out_451 * x_155
        out_451 = x_155 = None
        out_453 += out_443
        out_454 = out_453
        out_453 = out_443 = None
        out_455 = torch.nn.functional.relu(out_454, inplace=True)
        out_454 = None
        out_456 = torch.conv2d(
            out_455,
            l_self_modules_backbone_modules_layer3_modules_27_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_27_modules_conv1_parameters_weight_ = (
            None
        )
        out_457 = torch.nn.functional.batch_norm(
            out_456,
            l_self_modules_backbone_modules_layer3_modules_27_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_27_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_27_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_27_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_456 = l_self_modules_backbone_modules_layer3_modules_27_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_27_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_27_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_27_modules_bn1_parameters_bias_ = (None)
        out_458 = torch.nn.functional.relu(out_457, inplace=True)
        out_457 = None
        out_459 = torch.conv2d(
            out_458,
            l_self_modules_backbone_modules_layer3_modules_27_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_458 = l_self_modules_backbone_modules_layer3_modules_27_modules_conv2_parameters_weight_ = (None)
        out_460 = torch.nn.functional.batch_norm(
            out_459,
            l_self_modules_backbone_modules_layer3_modules_27_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_27_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_27_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_27_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_459 = l_self_modules_backbone_modules_layer3_modules_27_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_27_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_27_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_27_modules_bn2_parameters_bias_ = (None)
        out_461 = torch.nn.functional.relu(out_460, inplace=True)
        out_460 = None
        out_462 = torch.conv2d(
            out_461,
            l_self_modules_backbone_modules_layer3_modules_27_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_461 = l_self_modules_backbone_modules_layer3_modules_27_modules_conv3_parameters_weight_ = (None)
        out_463 = torch.nn.functional.batch_norm(
            out_462,
            l_self_modules_backbone_modules_layer3_modules_27_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_27_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_27_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_27_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_462 = l_self_modules_backbone_modules_layer3_modules_27_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_27_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_27_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_27_modules_bn3_parameters_bias_ = (None)
        out_464 = torch.nn.functional.adaptive_avg_pool2d(out_463, 1)
        x_156 = torch.conv2d(
            out_464,
            l_self_modules_backbone_modules_layer3_modules_27_modules_se_layer_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_27_modules_se_layer_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_464 = l_self_modules_backbone_modules_layer3_modules_27_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_27_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_157 = torch.nn.functional.relu(x_156, inplace=True)
        x_156 = None
        x_158 = torch.conv2d(
            x_157,
            l_self_modules_backbone_modules_layer3_modules_27_modules_se_layer_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_27_modules_se_layer_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_157 = l_self_modules_backbone_modules_layer3_modules_27_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_27_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = (None)
        x_159 = torch.sigmoid(x_158)
        x_158 = None
        out_465 = out_463 * x_159
        out_463 = x_159 = None
        out_465 += out_455
        out_466 = out_465
        out_465 = out_455 = None
        out_467 = torch.nn.functional.relu(out_466, inplace=True)
        out_466 = None
        out_468 = torch.conv2d(
            out_467,
            l_self_modules_backbone_modules_layer3_modules_28_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_28_modules_conv1_parameters_weight_ = (
            None
        )
        out_469 = torch.nn.functional.batch_norm(
            out_468,
            l_self_modules_backbone_modules_layer3_modules_28_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_28_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_28_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_28_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_468 = l_self_modules_backbone_modules_layer3_modules_28_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_28_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_28_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_28_modules_bn1_parameters_bias_ = (None)
        out_470 = torch.nn.functional.relu(out_469, inplace=True)
        out_469 = None
        out_471 = torch.conv2d(
            out_470,
            l_self_modules_backbone_modules_layer3_modules_28_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_470 = l_self_modules_backbone_modules_layer3_modules_28_modules_conv2_parameters_weight_ = (None)
        out_472 = torch.nn.functional.batch_norm(
            out_471,
            l_self_modules_backbone_modules_layer3_modules_28_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_28_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_28_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_28_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_471 = l_self_modules_backbone_modules_layer3_modules_28_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_28_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_28_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_28_modules_bn2_parameters_bias_ = (None)
        out_473 = torch.nn.functional.relu(out_472, inplace=True)
        out_472 = None
        out_474 = torch.conv2d(
            out_473,
            l_self_modules_backbone_modules_layer3_modules_28_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_473 = l_self_modules_backbone_modules_layer3_modules_28_modules_conv3_parameters_weight_ = (None)
        out_475 = torch.nn.functional.batch_norm(
            out_474,
            l_self_modules_backbone_modules_layer3_modules_28_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_28_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_28_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_28_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_474 = l_self_modules_backbone_modules_layer3_modules_28_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_28_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_28_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_28_modules_bn3_parameters_bias_ = (None)
        out_476 = torch.nn.functional.adaptive_avg_pool2d(out_475, 1)
        x_160 = torch.conv2d(
            out_476,
            l_self_modules_backbone_modules_layer3_modules_28_modules_se_layer_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_28_modules_se_layer_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_476 = l_self_modules_backbone_modules_layer3_modules_28_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_28_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_161 = torch.nn.functional.relu(x_160, inplace=True)
        x_160 = None
        x_162 = torch.conv2d(
            x_161,
            l_self_modules_backbone_modules_layer3_modules_28_modules_se_layer_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_28_modules_se_layer_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_161 = l_self_modules_backbone_modules_layer3_modules_28_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_28_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = (None)
        x_163 = torch.sigmoid(x_162)
        x_162 = None
        out_477 = out_475 * x_163
        out_475 = x_163 = None
        out_477 += out_467
        out_478 = out_477
        out_477 = out_467 = None
        out_479 = torch.nn.functional.relu(out_478, inplace=True)
        out_478 = None
        out_480 = torch.conv2d(
            out_479,
            l_self_modules_backbone_modules_layer3_modules_29_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_29_modules_conv1_parameters_weight_ = (
            None
        )
        out_481 = torch.nn.functional.batch_norm(
            out_480,
            l_self_modules_backbone_modules_layer3_modules_29_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_29_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_29_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_29_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_480 = l_self_modules_backbone_modules_layer3_modules_29_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_29_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_29_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_29_modules_bn1_parameters_bias_ = (None)
        out_482 = torch.nn.functional.relu(out_481, inplace=True)
        out_481 = None
        out_483 = torch.conv2d(
            out_482,
            l_self_modules_backbone_modules_layer3_modules_29_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_482 = l_self_modules_backbone_modules_layer3_modules_29_modules_conv2_parameters_weight_ = (None)
        out_484 = torch.nn.functional.batch_norm(
            out_483,
            l_self_modules_backbone_modules_layer3_modules_29_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_29_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_29_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_29_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_483 = l_self_modules_backbone_modules_layer3_modules_29_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_29_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_29_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_29_modules_bn2_parameters_bias_ = (None)
        out_485 = torch.nn.functional.relu(out_484, inplace=True)
        out_484 = None
        out_486 = torch.conv2d(
            out_485,
            l_self_modules_backbone_modules_layer3_modules_29_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_485 = l_self_modules_backbone_modules_layer3_modules_29_modules_conv3_parameters_weight_ = (None)
        out_487 = torch.nn.functional.batch_norm(
            out_486,
            l_self_modules_backbone_modules_layer3_modules_29_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_29_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_29_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_29_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_486 = l_self_modules_backbone_modules_layer3_modules_29_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_29_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_29_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_29_modules_bn3_parameters_bias_ = (None)
        out_488 = torch.nn.functional.adaptive_avg_pool2d(out_487, 1)
        x_164 = torch.conv2d(
            out_488,
            l_self_modules_backbone_modules_layer3_modules_29_modules_se_layer_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_29_modules_se_layer_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_488 = l_self_modules_backbone_modules_layer3_modules_29_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_29_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_165 = torch.nn.functional.relu(x_164, inplace=True)
        x_164 = None
        x_166 = torch.conv2d(
            x_165,
            l_self_modules_backbone_modules_layer3_modules_29_modules_se_layer_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_29_modules_se_layer_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_165 = l_self_modules_backbone_modules_layer3_modules_29_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_29_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = (None)
        x_167 = torch.sigmoid(x_166)
        x_166 = None
        out_489 = out_487 * x_167
        out_487 = x_167 = None
        out_489 += out_479
        out_490 = out_489
        out_489 = out_479 = None
        out_491 = torch.nn.functional.relu(out_490, inplace=True)
        out_490 = None
        out_492 = torch.conv2d(
            out_491,
            l_self_modules_backbone_modules_layer3_modules_30_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_30_modules_conv1_parameters_weight_ = (
            None
        )
        out_493 = torch.nn.functional.batch_norm(
            out_492,
            l_self_modules_backbone_modules_layer3_modules_30_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_30_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_30_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_30_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_492 = l_self_modules_backbone_modules_layer3_modules_30_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_30_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_30_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_30_modules_bn1_parameters_bias_ = (None)
        out_494 = torch.nn.functional.relu(out_493, inplace=True)
        out_493 = None
        out_495 = torch.conv2d(
            out_494,
            l_self_modules_backbone_modules_layer3_modules_30_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_494 = l_self_modules_backbone_modules_layer3_modules_30_modules_conv2_parameters_weight_ = (None)
        out_496 = torch.nn.functional.batch_norm(
            out_495,
            l_self_modules_backbone_modules_layer3_modules_30_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_30_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_30_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_30_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_495 = l_self_modules_backbone_modules_layer3_modules_30_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_30_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_30_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_30_modules_bn2_parameters_bias_ = (None)
        out_497 = torch.nn.functional.relu(out_496, inplace=True)
        out_496 = None
        out_498 = torch.conv2d(
            out_497,
            l_self_modules_backbone_modules_layer3_modules_30_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_497 = l_self_modules_backbone_modules_layer3_modules_30_modules_conv3_parameters_weight_ = (None)
        out_499 = torch.nn.functional.batch_norm(
            out_498,
            l_self_modules_backbone_modules_layer3_modules_30_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_30_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_30_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_30_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_498 = l_self_modules_backbone_modules_layer3_modules_30_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_30_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_30_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_30_modules_bn3_parameters_bias_ = (None)
        out_500 = torch.nn.functional.adaptive_avg_pool2d(out_499, 1)
        x_168 = torch.conv2d(
            out_500,
            l_self_modules_backbone_modules_layer3_modules_30_modules_se_layer_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_30_modules_se_layer_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_500 = l_self_modules_backbone_modules_layer3_modules_30_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_30_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_169 = torch.nn.functional.relu(x_168, inplace=True)
        x_168 = None
        x_170 = torch.conv2d(
            x_169,
            l_self_modules_backbone_modules_layer3_modules_30_modules_se_layer_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_30_modules_se_layer_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_169 = l_self_modules_backbone_modules_layer3_modules_30_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_30_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = (None)
        x_171 = torch.sigmoid(x_170)
        x_170 = None
        out_501 = out_499 * x_171
        out_499 = x_171 = None
        out_501 += out_491
        out_502 = out_501
        out_501 = out_491 = None
        out_503 = torch.nn.functional.relu(out_502, inplace=True)
        out_502 = None
        out_504 = torch.conv2d(
            out_503,
            l_self_modules_backbone_modules_layer3_modules_31_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_31_modules_conv1_parameters_weight_ = (
            None
        )
        out_505 = torch.nn.functional.batch_norm(
            out_504,
            l_self_modules_backbone_modules_layer3_modules_31_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_31_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_31_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_31_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_504 = l_self_modules_backbone_modules_layer3_modules_31_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_31_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_31_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_31_modules_bn1_parameters_bias_ = (None)
        out_506 = torch.nn.functional.relu(out_505, inplace=True)
        out_505 = None
        out_507 = torch.conv2d(
            out_506,
            l_self_modules_backbone_modules_layer3_modules_31_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_506 = l_self_modules_backbone_modules_layer3_modules_31_modules_conv2_parameters_weight_ = (None)
        out_508 = torch.nn.functional.batch_norm(
            out_507,
            l_self_modules_backbone_modules_layer3_modules_31_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_31_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_31_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_31_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_507 = l_self_modules_backbone_modules_layer3_modules_31_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_31_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_31_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_31_modules_bn2_parameters_bias_ = (None)
        out_509 = torch.nn.functional.relu(out_508, inplace=True)
        out_508 = None
        out_510 = torch.conv2d(
            out_509,
            l_self_modules_backbone_modules_layer3_modules_31_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_509 = l_self_modules_backbone_modules_layer3_modules_31_modules_conv3_parameters_weight_ = (None)
        out_511 = torch.nn.functional.batch_norm(
            out_510,
            l_self_modules_backbone_modules_layer3_modules_31_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_31_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_31_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_31_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_510 = l_self_modules_backbone_modules_layer3_modules_31_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_31_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_31_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_31_modules_bn3_parameters_bias_ = (None)
        out_512 = torch.nn.functional.adaptive_avg_pool2d(out_511, 1)
        x_172 = torch.conv2d(
            out_512,
            l_self_modules_backbone_modules_layer3_modules_31_modules_se_layer_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_31_modules_se_layer_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_512 = l_self_modules_backbone_modules_layer3_modules_31_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_31_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_173 = torch.nn.functional.relu(x_172, inplace=True)
        x_172 = None
        x_174 = torch.conv2d(
            x_173,
            l_self_modules_backbone_modules_layer3_modules_31_modules_se_layer_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_31_modules_se_layer_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_173 = l_self_modules_backbone_modules_layer3_modules_31_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_31_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = (None)
        x_175 = torch.sigmoid(x_174)
        x_174 = None
        out_513 = out_511 * x_175
        out_511 = x_175 = None
        out_513 += out_503
        out_514 = out_513
        out_513 = out_503 = None
        out_515 = torch.nn.functional.relu(out_514, inplace=True)
        out_514 = None
        out_516 = torch.conv2d(
            out_515,
            l_self_modules_backbone_modules_layer3_modules_32_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_32_modules_conv1_parameters_weight_ = (
            None
        )
        out_517 = torch.nn.functional.batch_norm(
            out_516,
            l_self_modules_backbone_modules_layer3_modules_32_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_32_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_32_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_32_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_516 = l_self_modules_backbone_modules_layer3_modules_32_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_32_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_32_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_32_modules_bn1_parameters_bias_ = (None)
        out_518 = torch.nn.functional.relu(out_517, inplace=True)
        out_517 = None
        out_519 = torch.conv2d(
            out_518,
            l_self_modules_backbone_modules_layer3_modules_32_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_518 = l_self_modules_backbone_modules_layer3_modules_32_modules_conv2_parameters_weight_ = (None)
        out_520 = torch.nn.functional.batch_norm(
            out_519,
            l_self_modules_backbone_modules_layer3_modules_32_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_32_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_32_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_32_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_519 = l_self_modules_backbone_modules_layer3_modules_32_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_32_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_32_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_32_modules_bn2_parameters_bias_ = (None)
        out_521 = torch.nn.functional.relu(out_520, inplace=True)
        out_520 = None
        out_522 = torch.conv2d(
            out_521,
            l_self_modules_backbone_modules_layer3_modules_32_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_521 = l_self_modules_backbone_modules_layer3_modules_32_modules_conv3_parameters_weight_ = (None)
        out_523 = torch.nn.functional.batch_norm(
            out_522,
            l_self_modules_backbone_modules_layer3_modules_32_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_32_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_32_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_32_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_522 = l_self_modules_backbone_modules_layer3_modules_32_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_32_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_32_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_32_modules_bn3_parameters_bias_ = (None)
        out_524 = torch.nn.functional.adaptive_avg_pool2d(out_523, 1)
        x_176 = torch.conv2d(
            out_524,
            l_self_modules_backbone_modules_layer3_modules_32_modules_se_layer_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_32_modules_se_layer_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_524 = l_self_modules_backbone_modules_layer3_modules_32_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_32_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_177 = torch.nn.functional.relu(x_176, inplace=True)
        x_176 = None
        x_178 = torch.conv2d(
            x_177,
            l_self_modules_backbone_modules_layer3_modules_32_modules_se_layer_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_32_modules_se_layer_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_177 = l_self_modules_backbone_modules_layer3_modules_32_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_32_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = (None)
        x_179 = torch.sigmoid(x_178)
        x_178 = None
        out_525 = out_523 * x_179
        out_523 = x_179 = None
        out_525 += out_515
        out_526 = out_525
        out_525 = out_515 = None
        out_527 = torch.nn.functional.relu(out_526, inplace=True)
        out_526 = None
        out_528 = torch.conv2d(
            out_527,
            l_self_modules_backbone_modules_layer3_modules_33_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_33_modules_conv1_parameters_weight_ = (
            None
        )
        out_529 = torch.nn.functional.batch_norm(
            out_528,
            l_self_modules_backbone_modules_layer3_modules_33_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_33_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_33_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_33_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_528 = l_self_modules_backbone_modules_layer3_modules_33_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_33_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_33_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_33_modules_bn1_parameters_bias_ = (None)
        out_530 = torch.nn.functional.relu(out_529, inplace=True)
        out_529 = None
        out_531 = torch.conv2d(
            out_530,
            l_self_modules_backbone_modules_layer3_modules_33_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_530 = l_self_modules_backbone_modules_layer3_modules_33_modules_conv2_parameters_weight_ = (None)
        out_532 = torch.nn.functional.batch_norm(
            out_531,
            l_self_modules_backbone_modules_layer3_modules_33_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_33_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_33_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_33_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_531 = l_self_modules_backbone_modules_layer3_modules_33_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_33_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_33_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_33_modules_bn2_parameters_bias_ = (None)
        out_533 = torch.nn.functional.relu(out_532, inplace=True)
        out_532 = None
        out_534 = torch.conv2d(
            out_533,
            l_self_modules_backbone_modules_layer3_modules_33_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_533 = l_self_modules_backbone_modules_layer3_modules_33_modules_conv3_parameters_weight_ = (None)
        out_535 = torch.nn.functional.batch_norm(
            out_534,
            l_self_modules_backbone_modules_layer3_modules_33_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_33_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_33_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_33_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_534 = l_self_modules_backbone_modules_layer3_modules_33_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_33_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_33_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_33_modules_bn3_parameters_bias_ = (None)
        out_536 = torch.nn.functional.adaptive_avg_pool2d(out_535, 1)
        x_180 = torch.conv2d(
            out_536,
            l_self_modules_backbone_modules_layer3_modules_33_modules_se_layer_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_33_modules_se_layer_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_536 = l_self_modules_backbone_modules_layer3_modules_33_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_33_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_181 = torch.nn.functional.relu(x_180, inplace=True)
        x_180 = None
        x_182 = torch.conv2d(
            x_181,
            l_self_modules_backbone_modules_layer3_modules_33_modules_se_layer_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_33_modules_se_layer_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_181 = l_self_modules_backbone_modules_layer3_modules_33_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_33_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = (None)
        x_183 = torch.sigmoid(x_182)
        x_182 = None
        out_537 = out_535 * x_183
        out_535 = x_183 = None
        out_537 += out_527
        out_538 = out_537
        out_537 = out_527 = None
        out_539 = torch.nn.functional.relu(out_538, inplace=True)
        out_538 = None
        out_540 = torch.conv2d(
            out_539,
            l_self_modules_backbone_modules_layer3_modules_34_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_34_modules_conv1_parameters_weight_ = (
            None
        )
        out_541 = torch.nn.functional.batch_norm(
            out_540,
            l_self_modules_backbone_modules_layer3_modules_34_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_34_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_34_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_34_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_540 = l_self_modules_backbone_modules_layer3_modules_34_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_34_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_34_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_34_modules_bn1_parameters_bias_ = (None)
        out_542 = torch.nn.functional.relu(out_541, inplace=True)
        out_541 = None
        out_543 = torch.conv2d(
            out_542,
            l_self_modules_backbone_modules_layer3_modules_34_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_542 = l_self_modules_backbone_modules_layer3_modules_34_modules_conv2_parameters_weight_ = (None)
        out_544 = torch.nn.functional.batch_norm(
            out_543,
            l_self_modules_backbone_modules_layer3_modules_34_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_34_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_34_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_34_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_543 = l_self_modules_backbone_modules_layer3_modules_34_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_34_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_34_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_34_modules_bn2_parameters_bias_ = (None)
        out_545 = torch.nn.functional.relu(out_544, inplace=True)
        out_544 = None
        out_546 = torch.conv2d(
            out_545,
            l_self_modules_backbone_modules_layer3_modules_34_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_545 = l_self_modules_backbone_modules_layer3_modules_34_modules_conv3_parameters_weight_ = (None)
        out_547 = torch.nn.functional.batch_norm(
            out_546,
            l_self_modules_backbone_modules_layer3_modules_34_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_34_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_34_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_34_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_546 = l_self_modules_backbone_modules_layer3_modules_34_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_34_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_34_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_34_modules_bn3_parameters_bias_ = (None)
        out_548 = torch.nn.functional.adaptive_avg_pool2d(out_547, 1)
        x_184 = torch.conv2d(
            out_548,
            l_self_modules_backbone_modules_layer3_modules_34_modules_se_layer_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_34_modules_se_layer_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_548 = l_self_modules_backbone_modules_layer3_modules_34_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_34_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_185 = torch.nn.functional.relu(x_184, inplace=True)
        x_184 = None
        x_186 = torch.conv2d(
            x_185,
            l_self_modules_backbone_modules_layer3_modules_34_modules_se_layer_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_34_modules_se_layer_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_185 = l_self_modules_backbone_modules_layer3_modules_34_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_34_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = (None)
        x_187 = torch.sigmoid(x_186)
        x_186 = None
        out_549 = out_547 * x_187
        out_547 = x_187 = None
        out_549 += out_539
        out_550 = out_549
        out_549 = out_539 = None
        out_551 = torch.nn.functional.relu(out_550, inplace=True)
        out_550 = None
        out_552 = torch.conv2d(
            out_551,
            l_self_modules_backbone_modules_layer3_modules_35_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_35_modules_conv1_parameters_weight_ = (
            None
        )
        out_553 = torch.nn.functional.batch_norm(
            out_552,
            l_self_modules_backbone_modules_layer3_modules_35_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_35_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_35_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_35_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_552 = l_self_modules_backbone_modules_layer3_modules_35_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_35_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_35_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_35_modules_bn1_parameters_bias_ = (None)
        out_554 = torch.nn.functional.relu(out_553, inplace=True)
        out_553 = None
        out_555 = torch.conv2d(
            out_554,
            l_self_modules_backbone_modules_layer3_modules_35_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_554 = l_self_modules_backbone_modules_layer3_modules_35_modules_conv2_parameters_weight_ = (None)
        out_556 = torch.nn.functional.batch_norm(
            out_555,
            l_self_modules_backbone_modules_layer3_modules_35_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_35_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_35_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_35_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_555 = l_self_modules_backbone_modules_layer3_modules_35_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_35_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_35_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_35_modules_bn2_parameters_bias_ = (None)
        out_557 = torch.nn.functional.relu(out_556, inplace=True)
        out_556 = None
        out_558 = torch.conv2d(
            out_557,
            l_self_modules_backbone_modules_layer3_modules_35_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_557 = l_self_modules_backbone_modules_layer3_modules_35_modules_conv3_parameters_weight_ = (None)
        out_559 = torch.nn.functional.batch_norm(
            out_558,
            l_self_modules_backbone_modules_layer3_modules_35_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_35_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_35_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_35_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_558 = l_self_modules_backbone_modules_layer3_modules_35_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_35_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_35_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_35_modules_bn3_parameters_bias_ = (None)
        out_560 = torch.nn.functional.adaptive_avg_pool2d(out_559, 1)
        x_188 = torch.conv2d(
            out_560,
            l_self_modules_backbone_modules_layer3_modules_35_modules_se_layer_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_35_modules_se_layer_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_560 = l_self_modules_backbone_modules_layer3_modules_35_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_35_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_189 = torch.nn.functional.relu(x_188, inplace=True)
        x_188 = None
        x_190 = torch.conv2d(
            x_189,
            l_self_modules_backbone_modules_layer3_modules_35_modules_se_layer_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_35_modules_se_layer_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_189 = l_self_modules_backbone_modules_layer3_modules_35_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_35_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = (None)
        x_191 = torch.sigmoid(x_190)
        x_190 = None
        out_561 = out_559 * x_191
        out_559 = x_191 = None
        out_561 += out_551
        out_562 = out_561
        out_561 = out_551 = None
        out_563 = torch.nn.functional.relu(out_562, inplace=True)
        out_562 = None
        out_564 = torch.conv2d(
            out_563,
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
        out_565 = torch.nn.functional.batch_norm(
            out_564,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_564 = l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_bias_ = (None)
        out_566 = torch.nn.functional.relu(out_565, inplace=True)
        out_565 = None
        out_567 = torch.conv2d(
            out_566,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        out_566 = l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_parameters_weight_ = (None)
        out_568 = torch.nn.functional.batch_norm(
            out_567,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_567 = l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_bias_ = (None)
        out_569 = torch.nn.functional.relu(out_568, inplace=True)
        out_568 = None
        out_570 = torch.conv2d(
            out_569,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_569 = l_self_modules_backbone_modules_layer4_modules_0_modules_conv3_parameters_weight_ = (None)
        out_571 = torch.nn.functional.batch_norm(
            out_570,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_570 = l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_bias_ = (None)
        out_572 = torch.nn.functional.adaptive_avg_pool2d(out_571, 1)
        x_192 = torch.conv2d(
            out_572,
            l_self_modules_backbone_modules_layer4_modules_0_modules_se_layer_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_se_layer_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_572 = l_self_modules_backbone_modules_layer4_modules_0_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_193 = torch.nn.functional.relu(x_192, inplace=True)
        x_192 = None
        x_194 = torch.conv2d(
            x_193,
            l_self_modules_backbone_modules_layer4_modules_0_modules_se_layer_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_se_layer_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_193 = l_self_modules_backbone_modules_layer4_modules_0_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = (None)
        x_195 = torch.sigmoid(x_194)
        x_194 = None
        out_573 = out_571 * x_195
        out_571 = x_195 = None
        input_7 = torch.conv2d(
            out_563,
            l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        out_563 = l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_8 = torch.nn.functional.batch_norm(
            input_7,
            l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_7 = l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        out_573 += input_8
        out_574 = out_573
        out_573 = input_8 = None
        out_575 = torch.nn.functional.relu(out_574, inplace=True)
        out_574 = None
        out_576 = torch.conv2d(
            out_575,
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
        out_577 = torch.nn.functional.batch_norm(
            out_576,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_576 = l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_bias_ = (None)
        out_578 = torch.nn.functional.relu(out_577, inplace=True)
        out_577 = None
        out_579 = torch.conv2d(
            out_578,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_578 = l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_parameters_weight_ = (None)
        out_580 = torch.nn.functional.batch_norm(
            out_579,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_579 = l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_bias_ = (None)
        out_581 = torch.nn.functional.relu(out_580, inplace=True)
        out_580 = None
        out_582 = torch.conv2d(
            out_581,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_581 = l_self_modules_backbone_modules_layer4_modules_1_modules_conv3_parameters_weight_ = (None)
        out_583 = torch.nn.functional.batch_norm(
            out_582,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_582 = l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_parameters_bias_ = (None)
        out_584 = torch.nn.functional.adaptive_avg_pool2d(out_583, 1)
        x_196 = torch.conv2d(
            out_584,
            l_self_modules_backbone_modules_layer4_modules_1_modules_se_layer_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_se_layer_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_584 = l_self_modules_backbone_modules_layer4_modules_1_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_1_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_197 = torch.nn.functional.relu(x_196, inplace=True)
        x_196 = None
        x_198 = torch.conv2d(
            x_197,
            l_self_modules_backbone_modules_layer4_modules_1_modules_se_layer_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_se_layer_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_197 = l_self_modules_backbone_modules_layer4_modules_1_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_1_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = (None)
        x_199 = torch.sigmoid(x_198)
        x_198 = None
        out_585 = out_583 * x_199
        out_583 = x_199 = None
        out_585 += out_575
        out_586 = out_585
        out_585 = out_575 = None
        out_587 = torch.nn.functional.relu(out_586, inplace=True)
        out_586 = None
        out_588 = torch.conv2d(
            out_587,
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
        out_589 = torch.nn.functional.batch_norm(
            out_588,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_588 = l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_parameters_bias_ = (None)
        out_590 = torch.nn.functional.relu(out_589, inplace=True)
        out_589 = None
        out_591 = torch.conv2d(
            out_590,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_590 = l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_parameters_weight_ = (None)
        out_592 = torch.nn.functional.batch_norm(
            out_591,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_591 = l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_parameters_bias_ = (None)
        out_593 = torch.nn.functional.relu(out_592, inplace=True)
        out_592 = None
        out_594 = torch.conv2d(
            out_593,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_593 = l_self_modules_backbone_modules_layer4_modules_2_modules_conv3_parameters_weight_ = (None)
        out_595 = torch.nn.functional.batch_norm(
            out_594,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_594 = l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_parameters_bias_ = (None)
        out_596 = torch.nn.functional.adaptive_avg_pool2d(out_595, 1)
        x_200 = torch.conv2d(
            out_596,
            l_self_modules_backbone_modules_layer4_modules_2_modules_se_layer_modules_conv1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_se_layer_modules_conv1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_596 = l_self_modules_backbone_modules_layer4_modules_2_modules_se_layer_modules_conv1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_2_modules_se_layer_modules_conv1_modules_conv_parameters_bias_ = (None)
        x_201 = torch.nn.functional.relu(x_200, inplace=True)
        x_200 = None
        x_202 = torch.conv2d(
            x_201,
            l_self_modules_backbone_modules_layer4_modules_2_modules_se_layer_modules_conv2_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_se_layer_modules_conv2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_201 = l_self_modules_backbone_modules_layer4_modules_2_modules_se_layer_modules_conv2_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_2_modules_se_layer_modules_conv2_modules_conv_parameters_bias_ = (None)
        x_203 = torch.sigmoid(x_202)
        x_202 = None
        out_597 = out_595 * x_203
        out_595 = x_203 = None
        out_597 += out_587
        out_598 = out_597
        out_597 = out_587 = None
        out_599 = torch.nn.functional.relu(out_598, inplace=True)
        out_598 = None
        input_9 = torch.conv_transpose2d(
            out_599,
            l_self_modules_head_modules_deconv_layers_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (0, 0),
            1,
            (1, 1),
        )
        out_599 = (
            l_self_modules_head_modules_deconv_layers_modules_0_parameters_weight_
        ) = None
        input_10 = torch.nn.functional.batch_norm(
            input_9,
            l_self_modules_head_modules_deconv_layers_modules_1_buffers_running_mean_,
            l_self_modules_head_modules_deconv_layers_modules_1_buffers_running_var_,
            l_self_modules_head_modules_deconv_layers_modules_1_parameters_weight_,
            l_self_modules_head_modules_deconv_layers_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_9 = (
            l_self_modules_head_modules_deconv_layers_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_1_buffers_running_var_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_1_parameters_weight_
        ) = l_self_modules_head_modules_deconv_layers_modules_1_parameters_bias_ = None
        input_11 = torch.nn.functional.relu(input_10, inplace=True)
        input_10 = None
        input_12 = torch.conv_transpose2d(
            input_11,
            l_self_modules_head_modules_deconv_layers_modules_3_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (0, 0),
            1,
            (1, 1),
        )
        input_11 = (
            l_self_modules_head_modules_deconv_layers_modules_3_parameters_weight_
        ) = None
        input_13 = torch.nn.functional.batch_norm(
            input_12,
            l_self_modules_head_modules_deconv_layers_modules_4_buffers_running_mean_,
            l_self_modules_head_modules_deconv_layers_modules_4_buffers_running_var_,
            l_self_modules_head_modules_deconv_layers_modules_4_parameters_weight_,
            l_self_modules_head_modules_deconv_layers_modules_4_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_12 = (
            l_self_modules_head_modules_deconv_layers_modules_4_buffers_running_mean_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_4_buffers_running_var_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_4_parameters_weight_
        ) = l_self_modules_head_modules_deconv_layers_modules_4_parameters_bias_ = None
        input_14 = torch.nn.functional.relu(input_13, inplace=True)
        input_13 = None
        input_15 = torch.conv_transpose2d(
            input_14,
            l_self_modules_head_modules_deconv_layers_modules_6_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (0, 0),
            1,
            (1, 1),
        )
        input_14 = (
            l_self_modules_head_modules_deconv_layers_modules_6_parameters_weight_
        ) = None
        input_16 = torch.nn.functional.batch_norm(
            input_15,
            l_self_modules_head_modules_deconv_layers_modules_7_buffers_running_mean_,
            l_self_modules_head_modules_deconv_layers_modules_7_buffers_running_var_,
            l_self_modules_head_modules_deconv_layers_modules_7_parameters_weight_,
            l_self_modules_head_modules_deconv_layers_modules_7_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_15 = (
            l_self_modules_head_modules_deconv_layers_modules_7_buffers_running_mean_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_7_buffers_running_var_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_7_parameters_weight_
        ) = l_self_modules_head_modules_deconv_layers_modules_7_parameters_bias_ = None
        input_17 = torch.nn.functional.relu(input_16, inplace=True)
        input_16 = None
        x_204 = torch.conv2d(
            input_17,
            l_self_modules_head_modules_final_layer_parameters_weight_,
            l_self_modules_head_modules_final_layer_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_17 = (
            l_self_modules_head_modules_final_layer_parameters_weight_
        ) = l_self_modules_head_modules_final_layer_parameters_bias_ = None
        return (x_204,)
