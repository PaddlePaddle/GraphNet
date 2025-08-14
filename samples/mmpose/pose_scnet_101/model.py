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
        L_self_modules_backbone_modules_layer1_modules_0_modules_k1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_k1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_k1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_k1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_k1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer1_modules_1_modules_k1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_k1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_1_modules_k1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_1_modules_k1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_k1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer1_modules_2_modules_k1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_k1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_2_modules_k1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_2_modules_k1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_k1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer2_modules_0_modules_k1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_k1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_k1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_k1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_k1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer2_modules_1_modules_k1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_k1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_1_modules_k1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_1_modules_k1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_k1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer2_modules_2_modules_k1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_k1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_2_modules_k1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_2_modules_k1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_k1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer2_modules_3_modules_k1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_k1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_3_modules_k1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_3_modules_k1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_k1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_3_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_3_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k2_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k2_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_0_modules_k1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_k1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_k1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_k1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_k1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_1_modules_k1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_k1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_1_modules_k1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_1_modules_k1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_k1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_2_modules_k1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_k1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_2_modules_k1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_2_modules_k1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_k1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_3_modules_k1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_k1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_3_modules_k1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_3_modules_k1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_k1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_3_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_3_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k2_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k2_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_4_modules_k1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_k1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_4_modules_k1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_4_modules_k1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_k1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_4_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_4_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k2_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k2_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_5_modules_k1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_k1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_5_modules_k1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_5_modules_k1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_k1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_5_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_5_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k2_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k2_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_6_modules_k1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_k1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_6_modules_k1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_6_modules_k1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_k1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_6_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_6_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k2_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k2_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_7_modules_k1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_7_modules_k1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_7_modules_k1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_7_modules_k1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_7_modules_k1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_7_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_7_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_7_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_7_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_7_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k2_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k2_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_8_modules_k1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_8_modules_k1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_8_modules_k1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_8_modules_k1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_8_modules_k1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_8_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_8_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_8_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_8_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_8_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k2_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k2_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_9_modules_k1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_9_modules_k1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_9_modules_k1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_9_modules_k1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_9_modules_k1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_9_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_9_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_9_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_9_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_9_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k2_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k2_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_10_modules_k1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_10_modules_k1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_10_modules_k1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_10_modules_k1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_10_modules_k1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_10_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_10_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_10_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_10_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_10_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k2_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k2_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_11_modules_k1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_11_modules_k1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_11_modules_k1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_11_modules_k1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_11_modules_k1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_11_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_11_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_11_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_11_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_11_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k2_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k2_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_12_modules_k1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_12_modules_k1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_12_modules_k1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_12_modules_k1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_12_modules_k1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_12_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_12_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_12_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_12_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_12_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k2_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k2_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_13_modules_k1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_13_modules_k1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_13_modules_k1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_13_modules_k1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_13_modules_k1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_13_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_13_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_13_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_13_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_13_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k2_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k2_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_14_modules_k1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_14_modules_k1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_14_modules_k1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_14_modules_k1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_14_modules_k1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_14_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_14_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_14_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_14_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_14_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k2_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k2_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_15_modules_k1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_15_modules_k1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_15_modules_k1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_15_modules_k1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_15_modules_k1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_15_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_15_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_15_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_15_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_15_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k2_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k2_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_16_modules_k1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_16_modules_k1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_16_modules_k1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_16_modules_k1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_16_modules_k1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_16_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_16_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_16_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_16_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_16_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k2_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k2_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_17_modules_k1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_17_modules_k1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_17_modules_k1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_17_modules_k1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_17_modules_k1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_17_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_17_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_17_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_17_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_17_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k2_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k2_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_18_modules_k1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_18_modules_k1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_18_modules_k1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_18_modules_k1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_18_modules_k1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_18_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_18_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_18_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_18_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_18_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k2_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k2_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_19_modules_k1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_19_modules_k1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_19_modules_k1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_19_modules_k1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_19_modules_k1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_19_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_19_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_19_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_19_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_19_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k2_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k2_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_20_modules_k1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_20_modules_k1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_20_modules_k1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_20_modules_k1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_20_modules_k1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_20_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_20_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_20_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_20_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_20_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k2_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k2_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_21_modules_k1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_21_modules_k1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_21_modules_k1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_21_modules_k1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_21_modules_k1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_21_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_21_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_21_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_21_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_21_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k2_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k2_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer3_modules_22_modules_k1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_22_modules_k1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_22_modules_k1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_22_modules_k1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_22_modules_k1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_22_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_22_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_22_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_22_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_22_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k2_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k2_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer4_modules_0_modules_k1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_k1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_k1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_k1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_k1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer4_modules_1_modules_k1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_k1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_1_modules_k1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_1_modules_k1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_k1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_layer4_modules_2_modules_k1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_k1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_2_modules_k1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_2_modules_k1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_k1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_backbone_modules_layer1_modules_0_modules_k1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_k1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_k1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_0_modules_k1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_0_modules_k1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_0_modules_k1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_0_modules_k1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_k1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_k1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_0_modules_k1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_0_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_0_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_0_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_0_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_0_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k2_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k2_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k2_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k2_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k4_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k4_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k4_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k4_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k4_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k4_modules_1_parameters_bias_
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
        l_self_modules_backbone_modules_layer1_modules_1_modules_k1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_1_modules_k1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_1_modules_k1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_1_modules_k1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_1_modules_k1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_1_modules_k1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_1_modules_k1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_1_modules_k1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_1_modules_k1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_1_modules_k1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_1_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_1_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_1_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_1_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_1_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k2_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k2_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k2_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k2_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k4_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k4_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k4_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k4_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k4_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k4_modules_1_parameters_bias_
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
        l_self_modules_backbone_modules_layer1_modules_2_modules_k1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_2_modules_k1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_2_modules_k1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_2_modules_k1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_2_modules_k1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_2_modules_k1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_2_modules_k1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_2_modules_k1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_2_modules_k1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_2_modules_k1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_2_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_2_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_2_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_2_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_2_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k2_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k2_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k2_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k2_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k4_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k4_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k4_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k4_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k4_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k4_modules_1_parameters_bias_
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
        l_self_modules_backbone_modules_layer2_modules_0_modules_k1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_k1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_k1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_0_modules_k1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_0_modules_k1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_0_modules_k1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_0_modules_k1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_k1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_k1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_0_modules_k1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_0_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_0_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_0_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k2_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k2_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k2_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k2_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k4_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k4_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k4_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k4_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k4_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k4_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_0_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_bias_
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
        l_self_modules_backbone_modules_layer2_modules_1_modules_k1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_k1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_k1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_1_modules_k1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_1_modules_k1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_1_modules_k1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_1_modules_k1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_k1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_k1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_1_modules_k1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_1_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_1_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_1_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k2_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k2_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k2_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k2_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k4_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k4_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k4_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k4_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k4_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k4_modules_1_parameters_bias_
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
        l_self_modules_backbone_modules_layer2_modules_2_modules_k1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_2_modules_k1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_2_modules_k1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_2_modules_k1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_2_modules_k1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_2_modules_k1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_2_modules_k1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_2_modules_k1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_2_modules_k1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_2_modules_k1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_2_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_2_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_2_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_2_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_2_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k2_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k2_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k2_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k2_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k4_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k4_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k4_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k4_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k4_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k4_modules_1_parameters_bias_
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
        l_self_modules_backbone_modules_layer2_modules_3_modules_k1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_3_modules_k1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_3_modules_k1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_3_modules_k1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_3_modules_k1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_3_modules_k1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_3_modules_k1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_3_modules_k1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_3_modules_k1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_3_modules_k1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_3_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_3_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_3_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_3_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_3_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k2_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k2_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k2_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k2_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k2_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k2_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k2_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k4_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k4_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k4_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k4_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k4_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k4_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k4_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k4_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k4_modules_1_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_0_modules_k1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_k1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_k1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_0_modules_k1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_0_modules_k1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_0_modules_k1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_0_modules_k1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_k1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_k1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_0_modules_k1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k2_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k2_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k2_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k2_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k4_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k4_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k4_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k4_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k4_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k4_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_0_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_1_modules_k1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_k1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_k1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_1_modules_k1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_1_modules_k1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_1_modules_k1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_1_modules_k1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_k1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_k1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_1_modules_k1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k2_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k2_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k2_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k2_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k4_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k4_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k4_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k4_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k4_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k4_modules_1_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_2_modules_k1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_k1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_k1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_2_modules_k1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_2_modules_k1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_2_modules_k1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_2_modules_k1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_k1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_k1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_2_modules_k1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_2_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_2_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_2_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k2_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k2_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k2_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k2_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k4_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k4_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k4_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k4_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k4_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k4_modules_1_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_3_modules_k1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_3_modules_k1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_3_modules_k1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_3_modules_k1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_3_modules_k1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_3_modules_k1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_3_modules_k1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_3_modules_k1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_3_modules_k1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_3_modules_k1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_3_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_3_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_3_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_3_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_3_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k2_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k2_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k2_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k2_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k2_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k2_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k2_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k4_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k4_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k4_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k4_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k4_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k4_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k4_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k4_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k4_modules_1_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_4_modules_k1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_4_modules_k1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_4_modules_k1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_4_modules_k1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_4_modules_k1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_4_modules_k1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_4_modules_k1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_4_modules_k1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_4_modules_k1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_4_modules_k1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_4_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_4_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_4_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_4_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_4_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k2_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k2_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k2_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k2_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k2_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k2_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k2_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k4_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k4_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k4_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k4_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k4_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k4_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k4_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k4_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k4_modules_1_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_5_modules_k1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_5_modules_k1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_5_modules_k1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_5_modules_k1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_5_modules_k1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_5_modules_k1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_5_modules_k1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_5_modules_k1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_5_modules_k1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_5_modules_k1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_5_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_5_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_5_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_5_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_5_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k2_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k2_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k2_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k2_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k2_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k2_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k2_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k4_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k4_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k4_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k4_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k4_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k4_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k4_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k4_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k4_modules_1_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_6_modules_k1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_6_modules_k1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_6_modules_k1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_6_modules_k1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_6_modules_k1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_6_modules_k1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_6_modules_k1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_6_modules_k1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_6_modules_k1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_6_modules_k1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_6_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_6_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_6_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_6_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_6_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_6_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_6_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_6_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_6_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k2_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k2_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k2_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k2_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k2_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k2_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k2_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k4_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k4_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k4_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k4_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k4_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k4_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k4_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k4_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k4_modules_1_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_7_modules_k1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_7_modules_k1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_7_modules_k1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_7_modules_k1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_7_modules_k1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_7_modules_k1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_7_modules_k1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_7_modules_k1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_7_modules_k1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_7_modules_k1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_7_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_7_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_7_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_7_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_7_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_7_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_7_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_7_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_7_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_7_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k2_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k2_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k2_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k2_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k2_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k2_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k2_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k4_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k4_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k4_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k4_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k4_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k4_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k4_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k4_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k4_modules_1_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_8_modules_k1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_8_modules_k1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_8_modules_k1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_8_modules_k1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_8_modules_k1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_8_modules_k1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_8_modules_k1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_8_modules_k1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_8_modules_k1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_8_modules_k1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_8_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_8_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_8_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_8_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_8_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_8_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_8_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_8_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_8_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_8_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k2_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k2_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k2_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k2_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k2_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k2_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k2_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k4_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k4_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k4_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k4_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k4_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k4_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k4_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k4_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k4_modules_1_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_9_modules_k1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_9_modules_k1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_9_modules_k1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_9_modules_k1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_9_modules_k1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_9_modules_k1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_9_modules_k1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_9_modules_k1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_9_modules_k1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_9_modules_k1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_9_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_9_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_9_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_9_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_9_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_9_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_9_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_9_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_9_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_9_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k2_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k2_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k2_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k2_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k2_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k2_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k2_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k4_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k4_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k4_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k4_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k4_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k4_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k4_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k4_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k4_modules_1_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_10_modules_k1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_10_modules_k1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_10_modules_k1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_10_modules_k1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_10_modules_k1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_10_modules_k1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_10_modules_k1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_10_modules_k1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_10_modules_k1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_10_modules_k1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_10_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_10_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_10_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_10_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_10_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_10_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_10_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_10_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_10_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_10_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k2_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k2_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k2_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k2_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k2_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k2_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k2_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k4_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k4_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k4_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k4_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k4_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k4_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k4_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k4_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k4_modules_1_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_11_modules_k1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_11_modules_k1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_11_modules_k1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_11_modules_k1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_11_modules_k1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_11_modules_k1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_11_modules_k1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_11_modules_k1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_11_modules_k1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_11_modules_k1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_11_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_11_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_11_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_11_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_11_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_11_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_11_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_11_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_11_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_11_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k2_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k2_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k2_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k2_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k2_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k2_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k2_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k4_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k4_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k4_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k4_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k4_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k4_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k4_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k4_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k4_modules_1_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_12_modules_k1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_12_modules_k1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_12_modules_k1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_12_modules_k1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_12_modules_k1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_12_modules_k1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_12_modules_k1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_12_modules_k1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_12_modules_k1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_12_modules_k1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_12_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_12_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_12_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_12_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_12_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_12_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_12_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_12_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_12_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_12_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k2_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k2_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k2_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k2_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k2_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k2_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k2_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k4_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k4_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k4_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k4_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k4_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k4_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k4_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k4_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k4_modules_1_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_13_modules_k1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_13_modules_k1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_13_modules_k1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_13_modules_k1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_13_modules_k1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_13_modules_k1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_13_modules_k1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_13_modules_k1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_13_modules_k1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_13_modules_k1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_13_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_13_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_13_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_13_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_13_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_13_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_13_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_13_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_13_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_13_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k2_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k2_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k2_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k2_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k2_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k2_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k2_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k4_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k4_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k4_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k4_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k4_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k4_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k4_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k4_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k4_modules_1_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_14_modules_k1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_14_modules_k1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_14_modules_k1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_14_modules_k1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_14_modules_k1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_14_modules_k1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_14_modules_k1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_14_modules_k1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_14_modules_k1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_14_modules_k1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_14_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_14_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_14_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_14_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_14_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_14_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_14_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_14_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_14_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_14_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k2_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k2_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k2_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k2_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k2_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k2_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k2_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k4_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k4_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k4_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k4_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k4_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k4_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k4_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k4_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k4_modules_1_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_15_modules_k1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_15_modules_k1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_15_modules_k1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_15_modules_k1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_15_modules_k1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_15_modules_k1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_15_modules_k1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_15_modules_k1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_15_modules_k1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_15_modules_k1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_15_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_15_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_15_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_15_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_15_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_15_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_15_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_15_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_15_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_15_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k2_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k2_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k2_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k2_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k2_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k2_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k2_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k4_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k4_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k4_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k4_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k4_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k4_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k4_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k4_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k4_modules_1_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_16_modules_k1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_16_modules_k1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_16_modules_k1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_16_modules_k1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_16_modules_k1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_16_modules_k1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_16_modules_k1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_16_modules_k1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_16_modules_k1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_16_modules_k1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_16_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_16_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_16_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_16_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_16_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_16_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_16_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_16_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_16_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_16_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k2_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k2_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k2_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k2_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k2_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k2_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k2_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k4_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k4_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k4_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k4_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k4_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k4_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k4_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k4_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k4_modules_1_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_17_modules_k1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_17_modules_k1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_17_modules_k1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_17_modules_k1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_17_modules_k1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_17_modules_k1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_17_modules_k1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_17_modules_k1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_17_modules_k1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_17_modules_k1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_17_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_17_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_17_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_17_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_17_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_17_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_17_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_17_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_17_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_17_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k2_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k2_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k2_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k2_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k2_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k2_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k2_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k4_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k4_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k4_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k4_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k4_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k4_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k4_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k4_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k4_modules_1_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_18_modules_k1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_18_modules_k1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_18_modules_k1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_18_modules_k1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_18_modules_k1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_18_modules_k1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_18_modules_k1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_18_modules_k1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_18_modules_k1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_18_modules_k1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_18_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_18_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_18_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_18_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_18_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_18_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_18_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_18_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_18_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_18_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k2_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k2_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k2_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k2_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k2_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k2_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k2_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k4_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k4_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k4_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k4_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k4_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k4_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k4_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k4_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k4_modules_1_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_19_modules_k1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_19_modules_k1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_19_modules_k1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_19_modules_k1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_19_modules_k1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_19_modules_k1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_19_modules_k1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_19_modules_k1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_19_modules_k1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_19_modules_k1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_19_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_19_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_19_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_19_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_19_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_19_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_19_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_19_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_19_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_19_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k2_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k2_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k2_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k2_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k2_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k2_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k2_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k4_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k4_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k4_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k4_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k4_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k4_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k4_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k4_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k4_modules_1_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_20_modules_k1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_20_modules_k1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_20_modules_k1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_20_modules_k1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_20_modules_k1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_20_modules_k1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_20_modules_k1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_20_modules_k1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_20_modules_k1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_20_modules_k1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_20_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_20_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_20_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_20_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_20_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_20_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_20_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_20_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_20_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_20_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k2_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k2_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k2_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k2_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k2_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k2_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k2_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k4_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k4_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k4_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k4_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k4_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k4_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k4_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k4_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k4_modules_1_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_21_modules_k1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_21_modules_k1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_21_modules_k1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_21_modules_k1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_21_modules_k1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_21_modules_k1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_21_modules_k1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_21_modules_k1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_21_modules_k1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_21_modules_k1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_21_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_21_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_21_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_21_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_21_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_21_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_21_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_21_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_21_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_21_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k2_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k2_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k2_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k2_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k2_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k2_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k2_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k4_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k4_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k4_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k4_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k4_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k4_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k4_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k4_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k4_modules_1_parameters_bias_
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
        l_self_modules_backbone_modules_layer3_modules_22_modules_k1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_22_modules_k1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_22_modules_k1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_22_modules_k1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_22_modules_k1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_22_modules_k1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_22_modules_k1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_22_modules_k1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_22_modules_k1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_22_modules_k1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_22_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_22_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_22_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_22_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_22_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_22_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_22_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_22_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_22_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_22_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k2_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k2_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k2_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k2_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k2_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k2_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k2_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k4_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k4_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k4_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k4_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k4_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k4_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k4_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k4_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k4_modules_1_parameters_bias_
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
        l_self_modules_backbone_modules_layer4_modules_0_modules_k1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_k1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_k1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_0_modules_k1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_0_modules_k1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_0_modules_k1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_0_modules_k1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_k1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_k1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_0_modules_k1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k2_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k2_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k2_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k2_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k4_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k4_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k4_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k4_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k4_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k4_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_0_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_bias_
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
        l_self_modules_backbone_modules_layer4_modules_1_modules_k1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_k1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_k1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_1_modules_k1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_1_modules_k1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_1_modules_k1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_1_modules_k1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_k1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_k1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_1_modules_k1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k2_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k2_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k2_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k2_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k4_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k4_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k4_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k4_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k4_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k4_modules_1_parameters_bias_
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
        l_self_modules_backbone_modules_layer4_modules_2_modules_k1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_k1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_k1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_2_modules_k1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_2_modules_k1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_2_modules_k1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_2_modules_k1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_k1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_k1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_2_modules_k1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_2_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_2_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_2_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k2_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k2_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k2_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k2_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k4_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k4_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k4_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k4_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k4_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k4_modules_1_parameters_bias_
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
        out_a = torch.conv2d(
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
        out_a_1 = torch.nn.functional.batch_norm(
            out_a,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_a = l_self_modules_backbone_modules_layer1_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_0_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_0_modules_bn1_parameters_bias_ = (None)
        out_a_2 = torch.nn.functional.relu(out_a_1, inplace=True)
        out_a_1 = None
        input_1 = torch.conv2d(
            out_a_2,
            l_self_modules_backbone_modules_layer1_modules_0_modules_k1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_a_2 = l_self_modules_backbone_modules_layer1_modules_0_modules_k1_modules_0_parameters_weight_ = (None)
        input_2 = torch.nn.functional.batch_norm(
            input_1,
            l_self_modules_backbone_modules_layer1_modules_0_modules_k1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_k1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_k1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_k1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_1 = l_self_modules_backbone_modules_layer1_modules_0_modules_k1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_0_modules_k1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_0_modules_k1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_0_modules_k1_modules_1_parameters_bias_ = (None)
        input_3 = torch.nn.functional.relu(input_2, inplace=True)
        input_2 = None
        out_b = torch.conv2d(
            x_3,
            l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_parameters_weight_ = (
            None
        )
        out_b_1 = torch.nn.functional.batch_norm(
            out_b,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_b = l_self_modules_backbone_modules_layer1_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_0_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_0_modules_bn2_parameters_bias_ = (None)
        out_b_2 = torch.nn.functional.relu(out_b_1, inplace=True)
        out_b_1 = None
        input_4 = torch._C._nn.avg_pool2d(out_b_2, 4, 4, 0, False, True, None)
        input_5 = torch.conv2d(
            input_4,
            l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k2_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_4 = l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k2_modules_1_parameters_weight_ = (None)
        input_6 = torch.nn.functional.batch_norm(
            input_5,
            l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k2_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k2_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_5 = l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k2_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k2_modules_2_parameters_bias_ = (None)
        interpolate = torch.nn.functional.interpolate(input_6, (64, 48))
        input_6 = None
        add = torch.add(out_b_2, interpolate)
        interpolate = None
        out = torch.sigmoid(add)
        add = None
        input_7 = torch.conv2d(
            out_b_2,
            l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_b_2 = l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k3_modules_0_parameters_weight_ = (None)
        input_8 = torch.nn.functional.batch_norm(
            input_7,
            l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_7 = l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k3_modules_1_parameters_bias_ = (None)
        out_1 = torch.mul(input_8, out)
        input_8 = out = None
        input_9 = torch.conv2d(
            out_1,
            l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k4_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_1 = l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k4_modules_0_parameters_weight_ = (None)
        input_10 = torch.nn.functional.batch_norm(
            input_9,
            l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k4_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_9 = l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k4_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k4_modules_1_parameters_bias_ = (None)
        input_11 = torch.nn.functional.relu(input_10, inplace=True)
        input_10 = None
        cat = torch.cat([input_3, input_11], dim=1)
        input_3 = input_11 = None
        out_2 = torch.conv2d(
            cat,
            l_self_modules_backbone_modules_layer1_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat = l_self_modules_backbone_modules_layer1_modules_0_modules_conv3_parameters_weight_ = (None)
        out_3 = torch.nn.functional.batch_norm(
            out_2,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_2 = l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_parameters_bias_ = (None)
        input_12 = torch.conv2d(
            x_3,
            l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_3 = l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_13 = torch.nn.functional.batch_norm(
            input_12,
            l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_12 = l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        out_3 += input_13
        out_4 = out_3
        out_3 = input_13 = None
        out_5 = torch.nn.functional.relu(out_4, inplace=True)
        out_4 = None
        out_a_3 = torch.conv2d(
            out_5,
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
        out_a_4 = torch.nn.functional.batch_norm(
            out_a_3,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_a_3 = l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_parameters_bias_ = (None)
        out_a_5 = torch.nn.functional.relu(out_a_4, inplace=True)
        out_a_4 = None
        input_14 = torch.conv2d(
            out_a_5,
            l_self_modules_backbone_modules_layer1_modules_1_modules_k1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_a_5 = l_self_modules_backbone_modules_layer1_modules_1_modules_k1_modules_0_parameters_weight_ = (None)
        input_15 = torch.nn.functional.batch_norm(
            input_14,
            l_self_modules_backbone_modules_layer1_modules_1_modules_k1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_k1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_k1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_k1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_14 = l_self_modules_backbone_modules_layer1_modules_1_modules_k1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_1_modules_k1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_1_modules_k1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_1_modules_k1_modules_1_parameters_bias_ = (None)
        input_16 = torch.nn.functional.relu(input_15, inplace=True)
        input_15 = None
        out_b_3 = torch.conv2d(
            out_5,
            l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_parameters_weight_ = (
            None
        )
        out_b_4 = torch.nn.functional.batch_norm(
            out_b_3,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_b_3 = l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_parameters_bias_ = (None)
        out_b_5 = torch.nn.functional.relu(out_b_4, inplace=True)
        out_b_4 = None
        input_17 = torch._C._nn.avg_pool2d(out_b_5, 4, 4, 0, False, True, None)
        input_18 = torch.conv2d(
            input_17,
            l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k2_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_17 = l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k2_modules_1_parameters_weight_ = (None)
        input_19 = torch.nn.functional.batch_norm(
            input_18,
            l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k2_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k2_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_18 = l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k2_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k2_modules_2_parameters_bias_ = (None)
        interpolate_1 = torch.nn.functional.interpolate(input_19, (64, 48))
        input_19 = None
        add_1 = torch.add(out_b_5, interpolate_1)
        interpolate_1 = None
        out_6 = torch.sigmoid(add_1)
        add_1 = None
        input_20 = torch.conv2d(
            out_b_5,
            l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_b_5 = l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k3_modules_0_parameters_weight_ = (None)
        input_21 = torch.nn.functional.batch_norm(
            input_20,
            l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_20 = l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k3_modules_1_parameters_bias_ = (None)
        out_7 = torch.mul(input_21, out_6)
        input_21 = out_6 = None
        input_22 = torch.conv2d(
            out_7,
            l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k4_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_7 = l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k4_modules_0_parameters_weight_ = (None)
        input_23 = torch.nn.functional.batch_norm(
            input_22,
            l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k4_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_22 = l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k4_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k4_modules_1_parameters_bias_ = (None)
        input_24 = torch.nn.functional.relu(input_23, inplace=True)
        input_23 = None
        cat_1 = torch.cat([input_16, input_24], dim=1)
        input_16 = input_24 = None
        out_8 = torch.conv2d(
            cat_1,
            l_self_modules_backbone_modules_layer1_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_1 = l_self_modules_backbone_modules_layer1_modules_1_modules_conv3_parameters_weight_ = (None)
        out_9 = torch.nn.functional.batch_norm(
            out_8,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_8 = l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_parameters_bias_ = (None)
        out_9 += out_5
        out_10 = out_9
        out_9 = out_5 = None
        out_11 = torch.nn.functional.relu(out_10, inplace=True)
        out_10 = None
        out_a_6 = torch.conv2d(
            out_11,
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
        out_a_7 = torch.nn.functional.batch_norm(
            out_a_6,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_a_6 = l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_parameters_bias_ = (None)
        out_a_8 = torch.nn.functional.relu(out_a_7, inplace=True)
        out_a_7 = None
        input_25 = torch.conv2d(
            out_a_8,
            l_self_modules_backbone_modules_layer1_modules_2_modules_k1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_a_8 = l_self_modules_backbone_modules_layer1_modules_2_modules_k1_modules_0_parameters_weight_ = (None)
        input_26 = torch.nn.functional.batch_norm(
            input_25,
            l_self_modules_backbone_modules_layer1_modules_2_modules_k1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_k1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_k1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_k1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_25 = l_self_modules_backbone_modules_layer1_modules_2_modules_k1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_2_modules_k1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_2_modules_k1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_2_modules_k1_modules_1_parameters_bias_ = (None)
        input_27 = torch.nn.functional.relu(input_26, inplace=True)
        input_26 = None
        out_b_6 = torch.conv2d(
            out_11,
            l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_parameters_weight_ = (
            None
        )
        out_b_7 = torch.nn.functional.batch_norm(
            out_b_6,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_b_6 = l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_parameters_bias_ = (None)
        out_b_8 = torch.nn.functional.relu(out_b_7, inplace=True)
        out_b_7 = None
        input_28 = torch._C._nn.avg_pool2d(out_b_8, 4, 4, 0, False, True, None)
        input_29 = torch.conv2d(
            input_28,
            l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k2_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_28 = l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k2_modules_1_parameters_weight_ = (None)
        input_30 = torch.nn.functional.batch_norm(
            input_29,
            l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k2_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k2_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_29 = l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k2_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k2_modules_2_parameters_bias_ = (None)
        interpolate_2 = torch.nn.functional.interpolate(input_30, (64, 48))
        input_30 = None
        add_2 = torch.add(out_b_8, interpolate_2)
        interpolate_2 = None
        out_12 = torch.sigmoid(add_2)
        add_2 = None
        input_31 = torch.conv2d(
            out_b_8,
            l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_b_8 = l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k3_modules_0_parameters_weight_ = (None)
        input_32 = torch.nn.functional.batch_norm(
            input_31,
            l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_31 = l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k3_modules_1_parameters_bias_ = (None)
        out_13 = torch.mul(input_32, out_12)
        input_32 = out_12 = None
        input_33 = torch.conv2d(
            out_13,
            l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k4_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_13 = l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k4_modules_0_parameters_weight_ = (None)
        input_34 = torch.nn.functional.batch_norm(
            input_33,
            l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k4_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_33 = l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k4_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k4_modules_1_parameters_bias_ = (None)
        input_35 = torch.nn.functional.relu(input_34, inplace=True)
        input_34 = None
        cat_2 = torch.cat([input_27, input_35], dim=1)
        input_27 = input_35 = None
        out_14 = torch.conv2d(
            cat_2,
            l_self_modules_backbone_modules_layer1_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_2 = l_self_modules_backbone_modules_layer1_modules_2_modules_conv3_parameters_weight_ = (None)
        out_15 = torch.nn.functional.batch_norm(
            out_14,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_14 = l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_parameters_bias_ = (None)
        out_15 += out_11
        out_16 = out_15
        out_15 = out_11 = None
        out_17 = torch.nn.functional.relu(out_16, inplace=True)
        out_16 = None
        out_a_9 = torch.conv2d(
            out_17,
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
        out_a_10 = torch.nn.functional.batch_norm(
            out_a_9,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_a_9 = l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_parameters_bias_ = (None)
        out_a_11 = torch.nn.functional.relu(out_a_10, inplace=True)
        out_a_10 = None
        input_36 = torch.conv2d(
            out_a_11,
            l_self_modules_backbone_modules_layer2_modules_0_modules_k1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        out_a_11 = l_self_modules_backbone_modules_layer2_modules_0_modules_k1_modules_0_parameters_weight_ = (None)
        input_37 = torch.nn.functional.batch_norm(
            input_36,
            l_self_modules_backbone_modules_layer2_modules_0_modules_k1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_k1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_k1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_k1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_36 = l_self_modules_backbone_modules_layer2_modules_0_modules_k1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_0_modules_k1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_0_modules_k1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_0_modules_k1_modules_1_parameters_bias_ = (None)
        input_38 = torch.nn.functional.relu(input_37, inplace=True)
        input_37 = None
        out_b_9 = torch.conv2d(
            out_17,
            l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_parameters_weight_ = (
            None
        )
        out_b_10 = torch.nn.functional.batch_norm(
            out_b_9,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_b_9 = l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_parameters_bias_ = (None)
        out_b_11 = torch.nn.functional.relu(out_b_10, inplace=True)
        out_b_10 = None
        input_39 = torch._C._nn.avg_pool2d(out_b_11, 4, 4, 0, False, True, None)
        input_40 = torch.conv2d(
            input_39,
            l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k2_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_39 = l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k2_modules_1_parameters_weight_ = (None)
        input_41 = torch.nn.functional.batch_norm(
            input_40,
            l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k2_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k2_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_40 = l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k2_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k2_modules_2_parameters_bias_ = (None)
        interpolate_3 = torch.nn.functional.interpolate(input_41, (64, 48))
        input_41 = None
        add_3 = torch.add(out_b_11, interpolate_3)
        interpolate_3 = None
        out_18 = torch.sigmoid(add_3)
        add_3 = None
        input_42 = torch.conv2d(
            out_b_11,
            l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_b_11 = l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k3_modules_0_parameters_weight_ = (None)
        input_43 = torch.nn.functional.batch_norm(
            input_42,
            l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_42 = l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k3_modules_1_parameters_bias_ = (None)
        out_19 = torch.mul(input_43, out_18)
        input_43 = out_18 = None
        input_44 = torch.conv2d(
            out_19,
            l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k4_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        out_19 = l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k4_modules_0_parameters_weight_ = (None)
        input_45 = torch.nn.functional.batch_norm(
            input_44,
            l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k4_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_44 = l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k4_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k4_modules_1_parameters_bias_ = (None)
        input_46 = torch.nn.functional.relu(input_45, inplace=True)
        input_45 = None
        cat_3 = torch.cat([input_38, input_46], dim=1)
        input_38 = input_46 = None
        out_20 = torch.conv2d(
            cat_3,
            l_self_modules_backbone_modules_layer2_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_3 = l_self_modules_backbone_modules_layer2_modules_0_modules_conv3_parameters_weight_ = (None)
        out_21 = torch.nn.functional.batch_norm(
            out_20,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_20 = l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_bias_ = (None)
        input_47 = torch.conv2d(
            out_17,
            l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        out_17 = l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_48 = torch.nn.functional.batch_norm(
            input_47,
            l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_47 = l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        out_21 += input_48
        out_22 = out_21
        out_21 = input_48 = None
        out_23 = torch.nn.functional.relu(out_22, inplace=True)
        out_22 = None
        out_a_12 = torch.conv2d(
            out_23,
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
        out_a_13 = torch.nn.functional.batch_norm(
            out_a_12,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_a_12 = l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_bias_ = (None)
        out_a_14 = torch.nn.functional.relu(out_a_13, inplace=True)
        out_a_13 = None
        input_49 = torch.conv2d(
            out_a_14,
            l_self_modules_backbone_modules_layer2_modules_1_modules_k1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_a_14 = l_self_modules_backbone_modules_layer2_modules_1_modules_k1_modules_0_parameters_weight_ = (None)
        input_50 = torch.nn.functional.batch_norm(
            input_49,
            l_self_modules_backbone_modules_layer2_modules_1_modules_k1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_k1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_k1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_k1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_49 = l_self_modules_backbone_modules_layer2_modules_1_modules_k1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_1_modules_k1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_1_modules_k1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_1_modules_k1_modules_1_parameters_bias_ = (None)
        input_51 = torch.nn.functional.relu(input_50, inplace=True)
        input_50 = None
        out_b_12 = torch.conv2d(
            out_23,
            l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_parameters_weight_ = (
            None
        )
        out_b_13 = torch.nn.functional.batch_norm(
            out_b_12,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_b_12 = l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_parameters_bias_ = (None)
        out_b_14 = torch.nn.functional.relu(out_b_13, inplace=True)
        out_b_13 = None
        input_52 = torch._C._nn.avg_pool2d(out_b_14, 4, 4, 0, False, True, None)
        input_53 = torch.conv2d(
            input_52,
            l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k2_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_52 = l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k2_modules_1_parameters_weight_ = (None)
        input_54 = torch.nn.functional.batch_norm(
            input_53,
            l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k2_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k2_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_53 = l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k2_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k2_modules_2_parameters_bias_ = (None)
        interpolate_4 = torch.nn.functional.interpolate(input_54, (32, 24))
        input_54 = None
        add_4 = torch.add(out_b_14, interpolate_4)
        interpolate_4 = None
        out_24 = torch.sigmoid(add_4)
        add_4 = None
        input_55 = torch.conv2d(
            out_b_14,
            l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_b_14 = l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k3_modules_0_parameters_weight_ = (None)
        input_56 = torch.nn.functional.batch_norm(
            input_55,
            l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_55 = l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k3_modules_1_parameters_bias_ = (None)
        out_25 = torch.mul(input_56, out_24)
        input_56 = out_24 = None
        input_57 = torch.conv2d(
            out_25,
            l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k4_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_25 = l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k4_modules_0_parameters_weight_ = (None)
        input_58 = torch.nn.functional.batch_norm(
            input_57,
            l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k4_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_57 = l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k4_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k4_modules_1_parameters_bias_ = (None)
        input_59 = torch.nn.functional.relu(input_58, inplace=True)
        input_58 = None
        cat_4 = torch.cat([input_51, input_59], dim=1)
        input_51 = input_59 = None
        out_26 = torch.conv2d(
            cat_4,
            l_self_modules_backbone_modules_layer2_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_4 = l_self_modules_backbone_modules_layer2_modules_1_modules_conv3_parameters_weight_ = (None)
        out_27 = torch.nn.functional.batch_norm(
            out_26,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_26 = l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_parameters_bias_ = (None)
        out_27 += out_23
        out_28 = out_27
        out_27 = out_23 = None
        out_29 = torch.nn.functional.relu(out_28, inplace=True)
        out_28 = None
        out_a_15 = torch.conv2d(
            out_29,
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
        out_a_16 = torch.nn.functional.batch_norm(
            out_a_15,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_a_15 = l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_parameters_bias_ = (None)
        out_a_17 = torch.nn.functional.relu(out_a_16, inplace=True)
        out_a_16 = None
        input_60 = torch.conv2d(
            out_a_17,
            l_self_modules_backbone_modules_layer2_modules_2_modules_k1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_a_17 = l_self_modules_backbone_modules_layer2_modules_2_modules_k1_modules_0_parameters_weight_ = (None)
        input_61 = torch.nn.functional.batch_norm(
            input_60,
            l_self_modules_backbone_modules_layer2_modules_2_modules_k1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_k1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_k1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_k1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_60 = l_self_modules_backbone_modules_layer2_modules_2_modules_k1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_2_modules_k1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_2_modules_k1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_2_modules_k1_modules_1_parameters_bias_ = (None)
        input_62 = torch.nn.functional.relu(input_61, inplace=True)
        input_61 = None
        out_b_15 = torch.conv2d(
            out_29,
            l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_parameters_weight_ = (
            None
        )
        out_b_16 = torch.nn.functional.batch_norm(
            out_b_15,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_b_15 = l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_parameters_bias_ = (None)
        out_b_17 = torch.nn.functional.relu(out_b_16, inplace=True)
        out_b_16 = None
        input_63 = torch._C._nn.avg_pool2d(out_b_17, 4, 4, 0, False, True, None)
        input_64 = torch.conv2d(
            input_63,
            l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k2_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_63 = l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k2_modules_1_parameters_weight_ = (None)
        input_65 = torch.nn.functional.batch_norm(
            input_64,
            l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k2_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k2_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_64 = l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k2_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k2_modules_2_parameters_bias_ = (None)
        interpolate_5 = torch.nn.functional.interpolate(input_65, (32, 24))
        input_65 = None
        add_5 = torch.add(out_b_17, interpolate_5)
        interpolate_5 = None
        out_30 = torch.sigmoid(add_5)
        add_5 = None
        input_66 = torch.conv2d(
            out_b_17,
            l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_b_17 = l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k3_modules_0_parameters_weight_ = (None)
        input_67 = torch.nn.functional.batch_norm(
            input_66,
            l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_66 = l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k3_modules_1_parameters_bias_ = (None)
        out_31 = torch.mul(input_67, out_30)
        input_67 = out_30 = None
        input_68 = torch.conv2d(
            out_31,
            l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k4_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_31 = l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k4_modules_0_parameters_weight_ = (None)
        input_69 = torch.nn.functional.batch_norm(
            input_68,
            l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k4_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_68 = l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k4_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k4_modules_1_parameters_bias_ = (None)
        input_70 = torch.nn.functional.relu(input_69, inplace=True)
        input_69 = None
        cat_5 = torch.cat([input_62, input_70], dim=1)
        input_62 = input_70 = None
        out_32 = torch.conv2d(
            cat_5,
            l_self_modules_backbone_modules_layer2_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_5 = l_self_modules_backbone_modules_layer2_modules_2_modules_conv3_parameters_weight_ = (None)
        out_33 = torch.nn.functional.batch_norm(
            out_32,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_32 = l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_parameters_bias_ = (None)
        out_33 += out_29
        out_34 = out_33
        out_33 = out_29 = None
        out_35 = torch.nn.functional.relu(out_34, inplace=True)
        out_34 = None
        out_a_18 = torch.conv2d(
            out_35,
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
        out_a_19 = torch.nn.functional.batch_norm(
            out_a_18,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_a_18 = l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_parameters_bias_ = (None)
        out_a_20 = torch.nn.functional.relu(out_a_19, inplace=True)
        out_a_19 = None
        input_71 = torch.conv2d(
            out_a_20,
            l_self_modules_backbone_modules_layer2_modules_3_modules_k1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_a_20 = l_self_modules_backbone_modules_layer2_modules_3_modules_k1_modules_0_parameters_weight_ = (None)
        input_72 = torch.nn.functional.batch_norm(
            input_71,
            l_self_modules_backbone_modules_layer2_modules_3_modules_k1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_k1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_k1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_k1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_71 = l_self_modules_backbone_modules_layer2_modules_3_modules_k1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_3_modules_k1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_3_modules_k1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_3_modules_k1_modules_1_parameters_bias_ = (None)
        input_73 = torch.nn.functional.relu(input_72, inplace=True)
        input_72 = None
        out_b_18 = torch.conv2d(
            out_35,
            l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_parameters_weight_ = (
            None
        )
        out_b_19 = torch.nn.functional.batch_norm(
            out_b_18,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_b_18 = l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_parameters_bias_ = (None)
        out_b_20 = torch.nn.functional.relu(out_b_19, inplace=True)
        out_b_19 = None
        input_74 = torch._C._nn.avg_pool2d(out_b_20, 4, 4, 0, False, True, None)
        input_75 = torch.conv2d(
            input_74,
            l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k2_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_74 = l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k2_modules_1_parameters_weight_ = (None)
        input_76 = torch.nn.functional.batch_norm(
            input_75,
            l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k2_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k2_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k2_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k2_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_75 = l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k2_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k2_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k2_modules_2_parameters_bias_ = (None)
        interpolate_6 = torch.nn.functional.interpolate(input_76, (32, 24))
        input_76 = None
        add_6 = torch.add(out_b_20, interpolate_6)
        interpolate_6 = None
        out_36 = torch.sigmoid(add_6)
        add_6 = None
        input_77 = torch.conv2d(
            out_b_20,
            l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_b_20 = l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k3_modules_0_parameters_weight_ = (None)
        input_78 = torch.nn.functional.batch_norm(
            input_77,
            l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_77 = l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k3_modules_1_parameters_bias_ = (None)
        out_37 = torch.mul(input_78, out_36)
        input_78 = out_36 = None
        input_79 = torch.conv2d(
            out_37,
            l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k4_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_37 = l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k4_modules_0_parameters_weight_ = (None)
        input_80 = torch.nn.functional.batch_norm(
            input_79,
            l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k4_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k4_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k4_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_79 = l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k4_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k4_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k4_modules_1_parameters_bias_ = (None)
        input_81 = torch.nn.functional.relu(input_80, inplace=True)
        input_80 = None
        cat_6 = torch.cat([input_73, input_81], dim=1)
        input_73 = input_81 = None
        out_38 = torch.conv2d(
            cat_6,
            l_self_modules_backbone_modules_layer2_modules_3_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_6 = l_self_modules_backbone_modules_layer2_modules_3_modules_conv3_parameters_weight_ = (None)
        out_39 = torch.nn.functional.batch_norm(
            out_38,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_38 = l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_parameters_bias_ = (None)
        out_39 += out_35
        out_40 = out_39
        out_39 = out_35 = None
        out_41 = torch.nn.functional.relu(out_40, inplace=True)
        out_40 = None
        out_a_21 = torch.conv2d(
            out_41,
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
        out_a_22 = torch.nn.functional.batch_norm(
            out_a_21,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_a_21 = l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_bias_ = (None)
        out_a_23 = torch.nn.functional.relu(out_a_22, inplace=True)
        out_a_22 = None
        input_82 = torch.conv2d(
            out_a_23,
            l_self_modules_backbone_modules_layer3_modules_0_modules_k1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        out_a_23 = l_self_modules_backbone_modules_layer3_modules_0_modules_k1_modules_0_parameters_weight_ = (None)
        input_83 = torch.nn.functional.batch_norm(
            input_82,
            l_self_modules_backbone_modules_layer3_modules_0_modules_k1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_k1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_k1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_k1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_82 = l_self_modules_backbone_modules_layer3_modules_0_modules_k1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_0_modules_k1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_0_modules_k1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_k1_modules_1_parameters_bias_ = (None)
        input_84 = torch.nn.functional.relu(input_83, inplace=True)
        input_83 = None
        out_b_21 = torch.conv2d(
            out_41,
            l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_parameters_weight_ = (
            None
        )
        out_b_22 = torch.nn.functional.batch_norm(
            out_b_21,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_b_21 = l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_bias_ = (None)
        out_b_23 = torch.nn.functional.relu(out_b_22, inplace=True)
        out_b_22 = None
        input_85 = torch._C._nn.avg_pool2d(out_b_23, 4, 4, 0, False, True, None)
        input_86 = torch.conv2d(
            input_85,
            l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k2_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_85 = l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k2_modules_1_parameters_weight_ = (None)
        input_87 = torch.nn.functional.batch_norm(
            input_86,
            l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k2_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k2_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_86 = l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k2_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k2_modules_2_parameters_bias_ = (None)
        interpolate_7 = torch.nn.functional.interpolate(input_87, (32, 24))
        input_87 = None
        add_7 = torch.add(out_b_23, interpolate_7)
        interpolate_7 = None
        out_42 = torch.sigmoid(add_7)
        add_7 = None
        input_88 = torch.conv2d(
            out_b_23,
            l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_b_23 = l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k3_modules_0_parameters_weight_ = (None)
        input_89 = torch.nn.functional.batch_norm(
            input_88,
            l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_88 = l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k3_modules_1_parameters_bias_ = (None)
        out_43 = torch.mul(input_89, out_42)
        input_89 = out_42 = None
        input_90 = torch.conv2d(
            out_43,
            l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k4_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        out_43 = l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k4_modules_0_parameters_weight_ = (None)
        input_91 = torch.nn.functional.batch_norm(
            input_90,
            l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k4_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_90 = l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k4_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k4_modules_1_parameters_bias_ = (None)
        input_92 = torch.nn.functional.relu(input_91, inplace=True)
        input_91 = None
        cat_7 = torch.cat([input_84, input_92], dim=1)
        input_84 = input_92 = None
        out_44 = torch.conv2d(
            cat_7,
            l_self_modules_backbone_modules_layer3_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_7 = l_self_modules_backbone_modules_layer3_modules_0_modules_conv3_parameters_weight_ = (None)
        out_45 = torch.nn.functional.batch_norm(
            out_44,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_44 = l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_bias_ = (None)
        input_93 = torch.conv2d(
            out_41,
            l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        out_41 = l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_94 = torch.nn.functional.batch_norm(
            input_93,
            l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_93 = l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        out_45 += input_94
        out_46 = out_45
        out_45 = input_94 = None
        out_47 = torch.nn.functional.relu(out_46, inplace=True)
        out_46 = None
        out_a_24 = torch.conv2d(
            out_47,
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
        out_a_25 = torch.nn.functional.batch_norm(
            out_a_24,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_a_24 = l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_bias_ = (None)
        out_a_26 = torch.nn.functional.relu(out_a_25, inplace=True)
        out_a_25 = None
        input_95 = torch.conv2d(
            out_a_26,
            l_self_modules_backbone_modules_layer3_modules_1_modules_k1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_a_26 = l_self_modules_backbone_modules_layer3_modules_1_modules_k1_modules_0_parameters_weight_ = (None)
        input_96 = torch.nn.functional.batch_norm(
            input_95,
            l_self_modules_backbone_modules_layer3_modules_1_modules_k1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_k1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_k1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_k1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_95 = l_self_modules_backbone_modules_layer3_modules_1_modules_k1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_1_modules_k1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_1_modules_k1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_1_modules_k1_modules_1_parameters_bias_ = (None)
        input_97 = torch.nn.functional.relu(input_96, inplace=True)
        input_96 = None
        out_b_24 = torch.conv2d(
            out_47,
            l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_parameters_weight_ = (
            None
        )
        out_b_25 = torch.nn.functional.batch_norm(
            out_b_24,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_b_24 = l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_bias_ = (None)
        out_b_26 = torch.nn.functional.relu(out_b_25, inplace=True)
        out_b_25 = None
        input_98 = torch._C._nn.avg_pool2d(out_b_26, 4, 4, 0, False, True, None)
        input_99 = torch.conv2d(
            input_98,
            l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k2_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_98 = l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k2_modules_1_parameters_weight_ = (None)
        input_100 = torch.nn.functional.batch_norm(
            input_99,
            l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k2_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k2_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_99 = l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k2_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k2_modules_2_parameters_bias_ = (None)
        interpolate_8 = torch.nn.functional.interpolate(input_100, (16, 12))
        input_100 = None
        add_8 = torch.add(out_b_26, interpolate_8)
        interpolate_8 = None
        out_48 = torch.sigmoid(add_8)
        add_8 = None
        input_101 = torch.conv2d(
            out_b_26,
            l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_b_26 = l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k3_modules_0_parameters_weight_ = (None)
        input_102 = torch.nn.functional.batch_norm(
            input_101,
            l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_101 = l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k3_modules_1_parameters_bias_ = (None)
        out_49 = torch.mul(input_102, out_48)
        input_102 = out_48 = None
        input_103 = torch.conv2d(
            out_49,
            l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k4_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_49 = l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k4_modules_0_parameters_weight_ = (None)
        input_104 = torch.nn.functional.batch_norm(
            input_103,
            l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k4_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_103 = l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k4_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k4_modules_1_parameters_bias_ = (None)
        input_105 = torch.nn.functional.relu(input_104, inplace=True)
        input_104 = None
        cat_8 = torch.cat([input_97, input_105], dim=1)
        input_97 = input_105 = None
        out_50 = torch.conv2d(
            cat_8,
            l_self_modules_backbone_modules_layer3_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_8 = l_self_modules_backbone_modules_layer3_modules_1_modules_conv3_parameters_weight_ = (None)
        out_51 = torch.nn.functional.batch_norm(
            out_50,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_50 = l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_parameters_bias_ = (None)
        out_51 += out_47
        out_52 = out_51
        out_51 = out_47 = None
        out_53 = torch.nn.functional.relu(out_52, inplace=True)
        out_52 = None
        out_a_27 = torch.conv2d(
            out_53,
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
        out_a_28 = torch.nn.functional.batch_norm(
            out_a_27,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_a_27 = l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_parameters_bias_ = (None)
        out_a_29 = torch.nn.functional.relu(out_a_28, inplace=True)
        out_a_28 = None
        input_106 = torch.conv2d(
            out_a_29,
            l_self_modules_backbone_modules_layer3_modules_2_modules_k1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_a_29 = l_self_modules_backbone_modules_layer3_modules_2_modules_k1_modules_0_parameters_weight_ = (None)
        input_107 = torch.nn.functional.batch_norm(
            input_106,
            l_self_modules_backbone_modules_layer3_modules_2_modules_k1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_k1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_k1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_k1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_106 = l_self_modules_backbone_modules_layer3_modules_2_modules_k1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_2_modules_k1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_2_modules_k1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_2_modules_k1_modules_1_parameters_bias_ = (None)
        input_108 = torch.nn.functional.relu(input_107, inplace=True)
        input_107 = None
        out_b_27 = torch.conv2d(
            out_53,
            l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_parameters_weight_ = (
            None
        )
        out_b_28 = torch.nn.functional.batch_norm(
            out_b_27,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_b_27 = l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_parameters_bias_ = (None)
        out_b_29 = torch.nn.functional.relu(out_b_28, inplace=True)
        out_b_28 = None
        input_109 = torch._C._nn.avg_pool2d(out_b_29, 4, 4, 0, False, True, None)
        input_110 = torch.conv2d(
            input_109,
            l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k2_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_109 = l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k2_modules_1_parameters_weight_ = (None)
        input_111 = torch.nn.functional.batch_norm(
            input_110,
            l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k2_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k2_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_110 = l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k2_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k2_modules_2_parameters_bias_ = (None)
        interpolate_9 = torch.nn.functional.interpolate(input_111, (16, 12))
        input_111 = None
        add_9 = torch.add(out_b_29, interpolate_9)
        interpolate_9 = None
        out_54 = torch.sigmoid(add_9)
        add_9 = None
        input_112 = torch.conv2d(
            out_b_29,
            l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_b_29 = l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k3_modules_0_parameters_weight_ = (None)
        input_113 = torch.nn.functional.batch_norm(
            input_112,
            l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_112 = l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k3_modules_1_parameters_bias_ = (None)
        out_55 = torch.mul(input_113, out_54)
        input_113 = out_54 = None
        input_114 = torch.conv2d(
            out_55,
            l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k4_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_55 = l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k4_modules_0_parameters_weight_ = (None)
        input_115 = torch.nn.functional.batch_norm(
            input_114,
            l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k4_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_114 = l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k4_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k4_modules_1_parameters_bias_ = (None)
        input_116 = torch.nn.functional.relu(input_115, inplace=True)
        input_115 = None
        cat_9 = torch.cat([input_108, input_116], dim=1)
        input_108 = input_116 = None
        out_56 = torch.conv2d(
            cat_9,
            l_self_modules_backbone_modules_layer3_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_9 = l_self_modules_backbone_modules_layer3_modules_2_modules_conv3_parameters_weight_ = (None)
        out_57 = torch.nn.functional.batch_norm(
            out_56,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_56 = l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_parameters_bias_ = (None)
        out_57 += out_53
        out_58 = out_57
        out_57 = out_53 = None
        out_59 = torch.nn.functional.relu(out_58, inplace=True)
        out_58 = None
        out_a_30 = torch.conv2d(
            out_59,
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
        out_a_31 = torch.nn.functional.batch_norm(
            out_a_30,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_a_30 = l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_parameters_bias_ = (None)
        out_a_32 = torch.nn.functional.relu(out_a_31, inplace=True)
        out_a_31 = None
        input_117 = torch.conv2d(
            out_a_32,
            l_self_modules_backbone_modules_layer3_modules_3_modules_k1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_a_32 = l_self_modules_backbone_modules_layer3_modules_3_modules_k1_modules_0_parameters_weight_ = (None)
        input_118 = torch.nn.functional.batch_norm(
            input_117,
            l_self_modules_backbone_modules_layer3_modules_3_modules_k1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_k1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_k1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_k1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_117 = l_self_modules_backbone_modules_layer3_modules_3_modules_k1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_3_modules_k1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_3_modules_k1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_3_modules_k1_modules_1_parameters_bias_ = (None)
        input_119 = torch.nn.functional.relu(input_118, inplace=True)
        input_118 = None
        out_b_30 = torch.conv2d(
            out_59,
            l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_parameters_weight_ = (
            None
        )
        out_b_31 = torch.nn.functional.batch_norm(
            out_b_30,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_b_30 = l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_parameters_bias_ = (None)
        out_b_32 = torch.nn.functional.relu(out_b_31, inplace=True)
        out_b_31 = None
        input_120 = torch._C._nn.avg_pool2d(out_b_32, 4, 4, 0, False, True, None)
        input_121 = torch.conv2d(
            input_120,
            l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k2_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_120 = l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k2_modules_1_parameters_weight_ = (None)
        input_122 = torch.nn.functional.batch_norm(
            input_121,
            l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k2_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k2_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k2_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k2_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_121 = l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k2_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k2_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k2_modules_2_parameters_bias_ = (None)
        interpolate_10 = torch.nn.functional.interpolate(input_122, (16, 12))
        input_122 = None
        add_10 = torch.add(out_b_32, interpolate_10)
        interpolate_10 = None
        out_60 = torch.sigmoid(add_10)
        add_10 = None
        input_123 = torch.conv2d(
            out_b_32,
            l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_b_32 = l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k3_modules_0_parameters_weight_ = (None)
        input_124 = torch.nn.functional.batch_norm(
            input_123,
            l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_123 = l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k3_modules_1_parameters_bias_ = (None)
        out_61 = torch.mul(input_124, out_60)
        input_124 = out_60 = None
        input_125 = torch.conv2d(
            out_61,
            l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k4_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_61 = l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k4_modules_0_parameters_weight_ = (None)
        input_126 = torch.nn.functional.batch_norm(
            input_125,
            l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k4_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k4_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k4_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_125 = l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k4_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k4_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k4_modules_1_parameters_bias_ = (None)
        input_127 = torch.nn.functional.relu(input_126, inplace=True)
        input_126 = None
        cat_10 = torch.cat([input_119, input_127], dim=1)
        input_119 = input_127 = None
        out_62 = torch.conv2d(
            cat_10,
            l_self_modules_backbone_modules_layer3_modules_3_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_10 = l_self_modules_backbone_modules_layer3_modules_3_modules_conv3_parameters_weight_ = (None)
        out_63 = torch.nn.functional.batch_norm(
            out_62,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_62 = l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_parameters_bias_ = (None)
        out_63 += out_59
        out_64 = out_63
        out_63 = out_59 = None
        out_65 = torch.nn.functional.relu(out_64, inplace=True)
        out_64 = None
        out_a_33 = torch.conv2d(
            out_65,
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
        out_a_34 = torch.nn.functional.batch_norm(
            out_a_33,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_a_33 = l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_parameters_bias_ = (None)
        out_a_35 = torch.nn.functional.relu(out_a_34, inplace=True)
        out_a_34 = None
        input_128 = torch.conv2d(
            out_a_35,
            l_self_modules_backbone_modules_layer3_modules_4_modules_k1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_a_35 = l_self_modules_backbone_modules_layer3_modules_4_modules_k1_modules_0_parameters_weight_ = (None)
        input_129 = torch.nn.functional.batch_norm(
            input_128,
            l_self_modules_backbone_modules_layer3_modules_4_modules_k1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_k1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_k1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_k1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_128 = l_self_modules_backbone_modules_layer3_modules_4_modules_k1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_4_modules_k1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_4_modules_k1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_4_modules_k1_modules_1_parameters_bias_ = (None)
        input_130 = torch.nn.functional.relu(input_129, inplace=True)
        input_129 = None
        out_b_33 = torch.conv2d(
            out_65,
            l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_parameters_weight_ = (
            None
        )
        out_b_34 = torch.nn.functional.batch_norm(
            out_b_33,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_b_33 = l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_parameters_bias_ = (None)
        out_b_35 = torch.nn.functional.relu(out_b_34, inplace=True)
        out_b_34 = None
        input_131 = torch._C._nn.avg_pool2d(out_b_35, 4, 4, 0, False, True, None)
        input_132 = torch.conv2d(
            input_131,
            l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k2_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_131 = l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k2_modules_1_parameters_weight_ = (None)
        input_133 = torch.nn.functional.batch_norm(
            input_132,
            l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k2_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k2_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k2_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k2_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_132 = l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k2_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k2_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k2_modules_2_parameters_bias_ = (None)
        interpolate_11 = torch.nn.functional.interpolate(input_133, (16, 12))
        input_133 = None
        add_11 = torch.add(out_b_35, interpolate_11)
        interpolate_11 = None
        out_66 = torch.sigmoid(add_11)
        add_11 = None
        input_134 = torch.conv2d(
            out_b_35,
            l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_b_35 = l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k3_modules_0_parameters_weight_ = (None)
        input_135 = torch.nn.functional.batch_norm(
            input_134,
            l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_134 = l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k3_modules_1_parameters_bias_ = (None)
        out_67 = torch.mul(input_135, out_66)
        input_135 = out_66 = None
        input_136 = torch.conv2d(
            out_67,
            l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k4_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_67 = l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k4_modules_0_parameters_weight_ = (None)
        input_137 = torch.nn.functional.batch_norm(
            input_136,
            l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k4_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k4_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k4_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_136 = l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k4_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k4_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k4_modules_1_parameters_bias_ = (None)
        input_138 = torch.nn.functional.relu(input_137, inplace=True)
        input_137 = None
        cat_11 = torch.cat([input_130, input_138], dim=1)
        input_130 = input_138 = None
        out_68 = torch.conv2d(
            cat_11,
            l_self_modules_backbone_modules_layer3_modules_4_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_11 = l_self_modules_backbone_modules_layer3_modules_4_modules_conv3_parameters_weight_ = (None)
        out_69 = torch.nn.functional.batch_norm(
            out_68,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_68 = l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_parameters_bias_ = (None)
        out_69 += out_65
        out_70 = out_69
        out_69 = out_65 = None
        out_71 = torch.nn.functional.relu(out_70, inplace=True)
        out_70 = None
        out_a_36 = torch.conv2d(
            out_71,
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
        out_a_37 = torch.nn.functional.batch_norm(
            out_a_36,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_a_36 = l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_parameters_bias_ = (None)
        out_a_38 = torch.nn.functional.relu(out_a_37, inplace=True)
        out_a_37 = None
        input_139 = torch.conv2d(
            out_a_38,
            l_self_modules_backbone_modules_layer3_modules_5_modules_k1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_a_38 = l_self_modules_backbone_modules_layer3_modules_5_modules_k1_modules_0_parameters_weight_ = (None)
        input_140 = torch.nn.functional.batch_norm(
            input_139,
            l_self_modules_backbone_modules_layer3_modules_5_modules_k1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_k1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_k1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_k1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_139 = l_self_modules_backbone_modules_layer3_modules_5_modules_k1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_5_modules_k1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_5_modules_k1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_5_modules_k1_modules_1_parameters_bias_ = (None)
        input_141 = torch.nn.functional.relu(input_140, inplace=True)
        input_140 = None
        out_b_36 = torch.conv2d(
            out_71,
            l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_parameters_weight_ = (
            None
        )
        out_b_37 = torch.nn.functional.batch_norm(
            out_b_36,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_b_36 = l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_parameters_bias_ = (None)
        out_b_38 = torch.nn.functional.relu(out_b_37, inplace=True)
        out_b_37 = None
        input_142 = torch._C._nn.avg_pool2d(out_b_38, 4, 4, 0, False, True, None)
        input_143 = torch.conv2d(
            input_142,
            l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k2_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_142 = l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k2_modules_1_parameters_weight_ = (None)
        input_144 = torch.nn.functional.batch_norm(
            input_143,
            l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k2_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k2_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k2_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k2_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_143 = l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k2_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k2_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k2_modules_2_parameters_bias_ = (None)
        interpolate_12 = torch.nn.functional.interpolate(input_144, (16, 12))
        input_144 = None
        add_12 = torch.add(out_b_38, interpolate_12)
        interpolate_12 = None
        out_72 = torch.sigmoid(add_12)
        add_12 = None
        input_145 = torch.conv2d(
            out_b_38,
            l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_b_38 = l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k3_modules_0_parameters_weight_ = (None)
        input_146 = torch.nn.functional.batch_norm(
            input_145,
            l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_145 = l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k3_modules_1_parameters_bias_ = (None)
        out_73 = torch.mul(input_146, out_72)
        input_146 = out_72 = None
        input_147 = torch.conv2d(
            out_73,
            l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k4_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_73 = l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k4_modules_0_parameters_weight_ = (None)
        input_148 = torch.nn.functional.batch_norm(
            input_147,
            l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k4_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k4_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k4_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_147 = l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k4_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k4_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k4_modules_1_parameters_bias_ = (None)
        input_149 = torch.nn.functional.relu(input_148, inplace=True)
        input_148 = None
        cat_12 = torch.cat([input_141, input_149], dim=1)
        input_141 = input_149 = None
        out_74 = torch.conv2d(
            cat_12,
            l_self_modules_backbone_modules_layer3_modules_5_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_12 = l_self_modules_backbone_modules_layer3_modules_5_modules_conv3_parameters_weight_ = (None)
        out_75 = torch.nn.functional.batch_norm(
            out_74,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_74 = l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_parameters_bias_ = (None)
        out_75 += out_71
        out_76 = out_75
        out_75 = out_71 = None
        out_77 = torch.nn.functional.relu(out_76, inplace=True)
        out_76 = None
        out_a_39 = torch.conv2d(
            out_77,
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
        out_a_40 = torch.nn.functional.batch_norm(
            out_a_39,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_a_39 = l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_6_modules_bn1_parameters_bias_ = (None)
        out_a_41 = torch.nn.functional.relu(out_a_40, inplace=True)
        out_a_40 = None
        input_150 = torch.conv2d(
            out_a_41,
            l_self_modules_backbone_modules_layer3_modules_6_modules_k1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_a_41 = l_self_modules_backbone_modules_layer3_modules_6_modules_k1_modules_0_parameters_weight_ = (None)
        input_151 = torch.nn.functional.batch_norm(
            input_150,
            l_self_modules_backbone_modules_layer3_modules_6_modules_k1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_k1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_k1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_k1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_150 = l_self_modules_backbone_modules_layer3_modules_6_modules_k1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_6_modules_k1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_6_modules_k1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_6_modules_k1_modules_1_parameters_bias_ = (None)
        input_152 = torch.nn.functional.relu(input_151, inplace=True)
        input_151 = None
        out_b_39 = torch.conv2d(
            out_77,
            l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_6_modules_conv2_parameters_weight_ = (
            None
        )
        out_b_40 = torch.nn.functional.batch_norm(
            out_b_39,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_b_39 = l_self_modules_backbone_modules_layer3_modules_6_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_6_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_6_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_6_modules_bn2_parameters_bias_ = (None)
        out_b_41 = torch.nn.functional.relu(out_b_40, inplace=True)
        out_b_40 = None
        input_153 = torch._C._nn.avg_pool2d(out_b_41, 4, 4, 0, False, True, None)
        input_154 = torch.conv2d(
            input_153,
            l_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k2_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_153 = l_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k2_modules_1_parameters_weight_ = (None)
        input_155 = torch.nn.functional.batch_norm(
            input_154,
            l_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k2_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k2_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k2_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k2_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_154 = l_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k2_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k2_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k2_modules_2_parameters_bias_ = (None)
        interpolate_13 = torch.nn.functional.interpolate(input_155, (16, 12))
        input_155 = None
        add_13 = torch.add(out_b_41, interpolate_13)
        interpolate_13 = None
        out_78 = torch.sigmoid(add_13)
        add_13 = None
        input_156 = torch.conv2d(
            out_b_41,
            l_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_b_41 = l_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k3_modules_0_parameters_weight_ = (None)
        input_157 = torch.nn.functional.batch_norm(
            input_156,
            l_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_156 = l_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k3_modules_1_parameters_bias_ = (None)
        out_79 = torch.mul(input_157, out_78)
        input_157 = out_78 = None
        input_158 = torch.conv2d(
            out_79,
            l_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k4_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_79 = l_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k4_modules_0_parameters_weight_ = (None)
        input_159 = torch.nn.functional.batch_norm(
            input_158,
            l_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k4_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k4_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k4_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_158 = l_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k4_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k4_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_6_modules_scconv_modules_k4_modules_1_parameters_bias_ = (None)
        input_160 = torch.nn.functional.relu(input_159, inplace=True)
        input_159 = None
        cat_13 = torch.cat([input_152, input_160], dim=1)
        input_152 = input_160 = None
        out_80 = torch.conv2d(
            cat_13,
            l_self_modules_backbone_modules_layer3_modules_6_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_13 = l_self_modules_backbone_modules_layer3_modules_6_modules_conv3_parameters_weight_ = (None)
        out_81 = torch.nn.functional.batch_norm(
            out_80,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_80 = l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_6_modules_bn3_parameters_bias_ = (None)
        out_81 += out_77
        out_82 = out_81
        out_81 = out_77 = None
        out_83 = torch.nn.functional.relu(out_82, inplace=True)
        out_82 = None
        out_a_42 = torch.conv2d(
            out_83,
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
        out_a_43 = torch.nn.functional.batch_norm(
            out_a_42,
            l_self_modules_backbone_modules_layer3_modules_7_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_a_42 = l_self_modules_backbone_modules_layer3_modules_7_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_7_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_7_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_7_modules_bn1_parameters_bias_ = (None)
        out_a_44 = torch.nn.functional.relu(out_a_43, inplace=True)
        out_a_43 = None
        input_161 = torch.conv2d(
            out_a_44,
            l_self_modules_backbone_modules_layer3_modules_7_modules_k1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_a_44 = l_self_modules_backbone_modules_layer3_modules_7_modules_k1_modules_0_parameters_weight_ = (None)
        input_162 = torch.nn.functional.batch_norm(
            input_161,
            l_self_modules_backbone_modules_layer3_modules_7_modules_k1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_k1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_k1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_k1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_161 = l_self_modules_backbone_modules_layer3_modules_7_modules_k1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_7_modules_k1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_7_modules_k1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_7_modules_k1_modules_1_parameters_bias_ = (None)
        input_163 = torch.nn.functional.relu(input_162, inplace=True)
        input_162 = None
        out_b_42 = torch.conv2d(
            out_83,
            l_self_modules_backbone_modules_layer3_modules_7_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_7_modules_conv2_parameters_weight_ = (
            None
        )
        out_b_43 = torch.nn.functional.batch_norm(
            out_b_42,
            l_self_modules_backbone_modules_layer3_modules_7_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_b_42 = l_self_modules_backbone_modules_layer3_modules_7_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_7_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_7_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_7_modules_bn2_parameters_bias_ = (None)
        out_b_44 = torch.nn.functional.relu(out_b_43, inplace=True)
        out_b_43 = None
        input_164 = torch._C._nn.avg_pool2d(out_b_44, 4, 4, 0, False, True, None)
        input_165 = torch.conv2d(
            input_164,
            l_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k2_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_164 = l_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k2_modules_1_parameters_weight_ = (None)
        input_166 = torch.nn.functional.batch_norm(
            input_165,
            l_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k2_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k2_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k2_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k2_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_165 = l_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k2_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k2_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k2_modules_2_parameters_bias_ = (None)
        interpolate_14 = torch.nn.functional.interpolate(input_166, (16, 12))
        input_166 = None
        add_14 = torch.add(out_b_44, interpolate_14)
        interpolate_14 = None
        out_84 = torch.sigmoid(add_14)
        add_14 = None
        input_167 = torch.conv2d(
            out_b_44,
            l_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_b_44 = l_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k3_modules_0_parameters_weight_ = (None)
        input_168 = torch.nn.functional.batch_norm(
            input_167,
            l_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_167 = l_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k3_modules_1_parameters_bias_ = (None)
        out_85 = torch.mul(input_168, out_84)
        input_168 = out_84 = None
        input_169 = torch.conv2d(
            out_85,
            l_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k4_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_85 = l_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k4_modules_0_parameters_weight_ = (None)
        input_170 = torch.nn.functional.batch_norm(
            input_169,
            l_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k4_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k4_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k4_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_169 = l_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k4_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k4_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_7_modules_scconv_modules_k4_modules_1_parameters_bias_ = (None)
        input_171 = torch.nn.functional.relu(input_170, inplace=True)
        input_170 = None
        cat_14 = torch.cat([input_163, input_171], dim=1)
        input_163 = input_171 = None
        out_86 = torch.conv2d(
            cat_14,
            l_self_modules_backbone_modules_layer3_modules_7_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_14 = l_self_modules_backbone_modules_layer3_modules_7_modules_conv3_parameters_weight_ = (None)
        out_87 = torch.nn.functional.batch_norm(
            out_86,
            l_self_modules_backbone_modules_layer3_modules_7_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_7_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_86 = l_self_modules_backbone_modules_layer3_modules_7_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_7_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_7_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_7_modules_bn3_parameters_bias_ = (None)
        out_87 += out_83
        out_88 = out_87
        out_87 = out_83 = None
        out_89 = torch.nn.functional.relu(out_88, inplace=True)
        out_88 = None
        out_a_45 = torch.conv2d(
            out_89,
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
        out_a_46 = torch.nn.functional.batch_norm(
            out_a_45,
            l_self_modules_backbone_modules_layer3_modules_8_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_a_45 = l_self_modules_backbone_modules_layer3_modules_8_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_8_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_8_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_8_modules_bn1_parameters_bias_ = (None)
        out_a_47 = torch.nn.functional.relu(out_a_46, inplace=True)
        out_a_46 = None
        input_172 = torch.conv2d(
            out_a_47,
            l_self_modules_backbone_modules_layer3_modules_8_modules_k1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_a_47 = l_self_modules_backbone_modules_layer3_modules_8_modules_k1_modules_0_parameters_weight_ = (None)
        input_173 = torch.nn.functional.batch_norm(
            input_172,
            l_self_modules_backbone_modules_layer3_modules_8_modules_k1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_k1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_k1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_k1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_172 = l_self_modules_backbone_modules_layer3_modules_8_modules_k1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_8_modules_k1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_8_modules_k1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_8_modules_k1_modules_1_parameters_bias_ = (None)
        input_174 = torch.nn.functional.relu(input_173, inplace=True)
        input_173 = None
        out_b_45 = torch.conv2d(
            out_89,
            l_self_modules_backbone_modules_layer3_modules_8_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_8_modules_conv2_parameters_weight_ = (
            None
        )
        out_b_46 = torch.nn.functional.batch_norm(
            out_b_45,
            l_self_modules_backbone_modules_layer3_modules_8_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_b_45 = l_self_modules_backbone_modules_layer3_modules_8_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_8_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_8_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_8_modules_bn2_parameters_bias_ = (None)
        out_b_47 = torch.nn.functional.relu(out_b_46, inplace=True)
        out_b_46 = None
        input_175 = torch._C._nn.avg_pool2d(out_b_47, 4, 4, 0, False, True, None)
        input_176 = torch.conv2d(
            input_175,
            l_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k2_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_175 = l_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k2_modules_1_parameters_weight_ = (None)
        input_177 = torch.nn.functional.batch_norm(
            input_176,
            l_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k2_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k2_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k2_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k2_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_176 = l_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k2_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k2_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k2_modules_2_parameters_bias_ = (None)
        interpolate_15 = torch.nn.functional.interpolate(input_177, (16, 12))
        input_177 = None
        add_15 = torch.add(out_b_47, interpolate_15)
        interpolate_15 = None
        out_90 = torch.sigmoid(add_15)
        add_15 = None
        input_178 = torch.conv2d(
            out_b_47,
            l_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_b_47 = l_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k3_modules_0_parameters_weight_ = (None)
        input_179 = torch.nn.functional.batch_norm(
            input_178,
            l_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_178 = l_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k3_modules_1_parameters_bias_ = (None)
        out_91 = torch.mul(input_179, out_90)
        input_179 = out_90 = None
        input_180 = torch.conv2d(
            out_91,
            l_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k4_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_91 = l_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k4_modules_0_parameters_weight_ = (None)
        input_181 = torch.nn.functional.batch_norm(
            input_180,
            l_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k4_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k4_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k4_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_180 = l_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k4_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k4_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_8_modules_scconv_modules_k4_modules_1_parameters_bias_ = (None)
        input_182 = torch.nn.functional.relu(input_181, inplace=True)
        input_181 = None
        cat_15 = torch.cat([input_174, input_182], dim=1)
        input_174 = input_182 = None
        out_92 = torch.conv2d(
            cat_15,
            l_self_modules_backbone_modules_layer3_modules_8_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_15 = l_self_modules_backbone_modules_layer3_modules_8_modules_conv3_parameters_weight_ = (None)
        out_93 = torch.nn.functional.batch_norm(
            out_92,
            l_self_modules_backbone_modules_layer3_modules_8_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_8_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_92 = l_self_modules_backbone_modules_layer3_modules_8_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_8_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_8_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_8_modules_bn3_parameters_bias_ = (None)
        out_93 += out_89
        out_94 = out_93
        out_93 = out_89 = None
        out_95 = torch.nn.functional.relu(out_94, inplace=True)
        out_94 = None
        out_a_48 = torch.conv2d(
            out_95,
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
        out_a_49 = torch.nn.functional.batch_norm(
            out_a_48,
            l_self_modules_backbone_modules_layer3_modules_9_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_a_48 = l_self_modules_backbone_modules_layer3_modules_9_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_9_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_9_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_9_modules_bn1_parameters_bias_ = (None)
        out_a_50 = torch.nn.functional.relu(out_a_49, inplace=True)
        out_a_49 = None
        input_183 = torch.conv2d(
            out_a_50,
            l_self_modules_backbone_modules_layer3_modules_9_modules_k1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_a_50 = l_self_modules_backbone_modules_layer3_modules_9_modules_k1_modules_0_parameters_weight_ = (None)
        input_184 = torch.nn.functional.batch_norm(
            input_183,
            l_self_modules_backbone_modules_layer3_modules_9_modules_k1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_k1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_k1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_k1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_183 = l_self_modules_backbone_modules_layer3_modules_9_modules_k1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_9_modules_k1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_9_modules_k1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_9_modules_k1_modules_1_parameters_bias_ = (None)
        input_185 = torch.nn.functional.relu(input_184, inplace=True)
        input_184 = None
        out_b_48 = torch.conv2d(
            out_95,
            l_self_modules_backbone_modules_layer3_modules_9_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_9_modules_conv2_parameters_weight_ = (
            None
        )
        out_b_49 = torch.nn.functional.batch_norm(
            out_b_48,
            l_self_modules_backbone_modules_layer3_modules_9_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_b_48 = l_self_modules_backbone_modules_layer3_modules_9_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_9_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_9_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_9_modules_bn2_parameters_bias_ = (None)
        out_b_50 = torch.nn.functional.relu(out_b_49, inplace=True)
        out_b_49 = None
        input_186 = torch._C._nn.avg_pool2d(out_b_50, 4, 4, 0, False, True, None)
        input_187 = torch.conv2d(
            input_186,
            l_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k2_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_186 = l_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k2_modules_1_parameters_weight_ = (None)
        input_188 = torch.nn.functional.batch_norm(
            input_187,
            l_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k2_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k2_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k2_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k2_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_187 = l_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k2_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k2_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k2_modules_2_parameters_bias_ = (None)
        interpolate_16 = torch.nn.functional.interpolate(input_188, (16, 12))
        input_188 = None
        add_16 = torch.add(out_b_50, interpolate_16)
        interpolate_16 = None
        out_96 = torch.sigmoid(add_16)
        add_16 = None
        input_189 = torch.conv2d(
            out_b_50,
            l_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_b_50 = l_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k3_modules_0_parameters_weight_ = (None)
        input_190 = torch.nn.functional.batch_norm(
            input_189,
            l_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_189 = l_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k3_modules_1_parameters_bias_ = (None)
        out_97 = torch.mul(input_190, out_96)
        input_190 = out_96 = None
        input_191 = torch.conv2d(
            out_97,
            l_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k4_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_97 = l_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k4_modules_0_parameters_weight_ = (None)
        input_192 = torch.nn.functional.batch_norm(
            input_191,
            l_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k4_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k4_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k4_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_191 = l_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k4_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k4_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_9_modules_scconv_modules_k4_modules_1_parameters_bias_ = (None)
        input_193 = torch.nn.functional.relu(input_192, inplace=True)
        input_192 = None
        cat_16 = torch.cat([input_185, input_193], dim=1)
        input_185 = input_193 = None
        out_98 = torch.conv2d(
            cat_16,
            l_self_modules_backbone_modules_layer3_modules_9_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_16 = l_self_modules_backbone_modules_layer3_modules_9_modules_conv3_parameters_weight_ = (None)
        out_99 = torch.nn.functional.batch_norm(
            out_98,
            l_self_modules_backbone_modules_layer3_modules_9_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_9_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_98 = l_self_modules_backbone_modules_layer3_modules_9_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_9_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_9_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_9_modules_bn3_parameters_bias_ = (None)
        out_99 += out_95
        out_100 = out_99
        out_99 = out_95 = None
        out_101 = torch.nn.functional.relu(out_100, inplace=True)
        out_100 = None
        out_a_51 = torch.conv2d(
            out_101,
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
        out_a_52 = torch.nn.functional.batch_norm(
            out_a_51,
            l_self_modules_backbone_modules_layer3_modules_10_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_a_51 = l_self_modules_backbone_modules_layer3_modules_10_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_10_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_10_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_10_modules_bn1_parameters_bias_ = (None)
        out_a_53 = torch.nn.functional.relu(out_a_52, inplace=True)
        out_a_52 = None
        input_194 = torch.conv2d(
            out_a_53,
            l_self_modules_backbone_modules_layer3_modules_10_modules_k1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_a_53 = l_self_modules_backbone_modules_layer3_modules_10_modules_k1_modules_0_parameters_weight_ = (None)
        input_195 = torch.nn.functional.batch_norm(
            input_194,
            l_self_modules_backbone_modules_layer3_modules_10_modules_k1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_k1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_k1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_k1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_194 = l_self_modules_backbone_modules_layer3_modules_10_modules_k1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_10_modules_k1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_10_modules_k1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_10_modules_k1_modules_1_parameters_bias_ = (None)
        input_196 = torch.nn.functional.relu(input_195, inplace=True)
        input_195 = None
        out_b_51 = torch.conv2d(
            out_101,
            l_self_modules_backbone_modules_layer3_modules_10_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_10_modules_conv2_parameters_weight_ = (
            None
        )
        out_b_52 = torch.nn.functional.batch_norm(
            out_b_51,
            l_self_modules_backbone_modules_layer3_modules_10_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_b_51 = l_self_modules_backbone_modules_layer3_modules_10_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_10_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_10_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_10_modules_bn2_parameters_bias_ = (None)
        out_b_53 = torch.nn.functional.relu(out_b_52, inplace=True)
        out_b_52 = None
        input_197 = torch._C._nn.avg_pool2d(out_b_53, 4, 4, 0, False, True, None)
        input_198 = torch.conv2d(
            input_197,
            l_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k2_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_197 = l_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k2_modules_1_parameters_weight_ = (None)
        input_199 = torch.nn.functional.batch_norm(
            input_198,
            l_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k2_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k2_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k2_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k2_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_198 = l_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k2_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k2_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k2_modules_2_parameters_bias_ = (None)
        interpolate_17 = torch.nn.functional.interpolate(input_199, (16, 12))
        input_199 = None
        add_17 = torch.add(out_b_53, interpolate_17)
        interpolate_17 = None
        out_102 = torch.sigmoid(add_17)
        add_17 = None
        input_200 = torch.conv2d(
            out_b_53,
            l_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_b_53 = l_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k3_modules_0_parameters_weight_ = (None)
        input_201 = torch.nn.functional.batch_norm(
            input_200,
            l_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_200 = l_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k3_modules_1_parameters_bias_ = (None)
        out_103 = torch.mul(input_201, out_102)
        input_201 = out_102 = None
        input_202 = torch.conv2d(
            out_103,
            l_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k4_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_103 = l_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k4_modules_0_parameters_weight_ = (None)
        input_203 = torch.nn.functional.batch_norm(
            input_202,
            l_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k4_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k4_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k4_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_202 = l_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k4_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k4_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_10_modules_scconv_modules_k4_modules_1_parameters_bias_ = (None)
        input_204 = torch.nn.functional.relu(input_203, inplace=True)
        input_203 = None
        cat_17 = torch.cat([input_196, input_204], dim=1)
        input_196 = input_204 = None
        out_104 = torch.conv2d(
            cat_17,
            l_self_modules_backbone_modules_layer3_modules_10_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_17 = l_self_modules_backbone_modules_layer3_modules_10_modules_conv3_parameters_weight_ = (None)
        out_105 = torch.nn.functional.batch_norm(
            out_104,
            l_self_modules_backbone_modules_layer3_modules_10_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_10_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_104 = l_self_modules_backbone_modules_layer3_modules_10_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_10_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_10_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_10_modules_bn3_parameters_bias_ = (None)
        out_105 += out_101
        out_106 = out_105
        out_105 = out_101 = None
        out_107 = torch.nn.functional.relu(out_106, inplace=True)
        out_106 = None
        out_a_54 = torch.conv2d(
            out_107,
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
        out_a_55 = torch.nn.functional.batch_norm(
            out_a_54,
            l_self_modules_backbone_modules_layer3_modules_11_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_a_54 = l_self_modules_backbone_modules_layer3_modules_11_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_11_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_11_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_11_modules_bn1_parameters_bias_ = (None)
        out_a_56 = torch.nn.functional.relu(out_a_55, inplace=True)
        out_a_55 = None
        input_205 = torch.conv2d(
            out_a_56,
            l_self_modules_backbone_modules_layer3_modules_11_modules_k1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_a_56 = l_self_modules_backbone_modules_layer3_modules_11_modules_k1_modules_0_parameters_weight_ = (None)
        input_206 = torch.nn.functional.batch_norm(
            input_205,
            l_self_modules_backbone_modules_layer3_modules_11_modules_k1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_k1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_k1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_k1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_205 = l_self_modules_backbone_modules_layer3_modules_11_modules_k1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_11_modules_k1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_11_modules_k1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_11_modules_k1_modules_1_parameters_bias_ = (None)
        input_207 = torch.nn.functional.relu(input_206, inplace=True)
        input_206 = None
        out_b_54 = torch.conv2d(
            out_107,
            l_self_modules_backbone_modules_layer3_modules_11_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_11_modules_conv2_parameters_weight_ = (
            None
        )
        out_b_55 = torch.nn.functional.batch_norm(
            out_b_54,
            l_self_modules_backbone_modules_layer3_modules_11_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_b_54 = l_self_modules_backbone_modules_layer3_modules_11_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_11_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_11_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_11_modules_bn2_parameters_bias_ = (None)
        out_b_56 = torch.nn.functional.relu(out_b_55, inplace=True)
        out_b_55 = None
        input_208 = torch._C._nn.avg_pool2d(out_b_56, 4, 4, 0, False, True, None)
        input_209 = torch.conv2d(
            input_208,
            l_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k2_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_208 = l_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k2_modules_1_parameters_weight_ = (None)
        input_210 = torch.nn.functional.batch_norm(
            input_209,
            l_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k2_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k2_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k2_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k2_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_209 = l_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k2_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k2_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k2_modules_2_parameters_bias_ = (None)
        interpolate_18 = torch.nn.functional.interpolate(input_210, (16, 12))
        input_210 = None
        add_18 = torch.add(out_b_56, interpolate_18)
        interpolate_18 = None
        out_108 = torch.sigmoid(add_18)
        add_18 = None
        input_211 = torch.conv2d(
            out_b_56,
            l_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_b_56 = l_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k3_modules_0_parameters_weight_ = (None)
        input_212 = torch.nn.functional.batch_norm(
            input_211,
            l_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_211 = l_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k3_modules_1_parameters_bias_ = (None)
        out_109 = torch.mul(input_212, out_108)
        input_212 = out_108 = None
        input_213 = torch.conv2d(
            out_109,
            l_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k4_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_109 = l_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k4_modules_0_parameters_weight_ = (None)
        input_214 = torch.nn.functional.batch_norm(
            input_213,
            l_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k4_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k4_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k4_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_213 = l_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k4_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k4_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_11_modules_scconv_modules_k4_modules_1_parameters_bias_ = (None)
        input_215 = torch.nn.functional.relu(input_214, inplace=True)
        input_214 = None
        cat_18 = torch.cat([input_207, input_215], dim=1)
        input_207 = input_215 = None
        out_110 = torch.conv2d(
            cat_18,
            l_self_modules_backbone_modules_layer3_modules_11_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_18 = l_self_modules_backbone_modules_layer3_modules_11_modules_conv3_parameters_weight_ = (None)
        out_111 = torch.nn.functional.batch_norm(
            out_110,
            l_self_modules_backbone_modules_layer3_modules_11_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_11_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_110 = l_self_modules_backbone_modules_layer3_modules_11_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_11_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_11_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_11_modules_bn3_parameters_bias_ = (None)
        out_111 += out_107
        out_112 = out_111
        out_111 = out_107 = None
        out_113 = torch.nn.functional.relu(out_112, inplace=True)
        out_112 = None
        out_a_57 = torch.conv2d(
            out_113,
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
        out_a_58 = torch.nn.functional.batch_norm(
            out_a_57,
            l_self_modules_backbone_modules_layer3_modules_12_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_a_57 = l_self_modules_backbone_modules_layer3_modules_12_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_12_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_12_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_12_modules_bn1_parameters_bias_ = (None)
        out_a_59 = torch.nn.functional.relu(out_a_58, inplace=True)
        out_a_58 = None
        input_216 = torch.conv2d(
            out_a_59,
            l_self_modules_backbone_modules_layer3_modules_12_modules_k1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_a_59 = l_self_modules_backbone_modules_layer3_modules_12_modules_k1_modules_0_parameters_weight_ = (None)
        input_217 = torch.nn.functional.batch_norm(
            input_216,
            l_self_modules_backbone_modules_layer3_modules_12_modules_k1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_k1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_k1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_k1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_216 = l_self_modules_backbone_modules_layer3_modules_12_modules_k1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_12_modules_k1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_12_modules_k1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_12_modules_k1_modules_1_parameters_bias_ = (None)
        input_218 = torch.nn.functional.relu(input_217, inplace=True)
        input_217 = None
        out_b_57 = torch.conv2d(
            out_113,
            l_self_modules_backbone_modules_layer3_modules_12_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_12_modules_conv2_parameters_weight_ = (
            None
        )
        out_b_58 = torch.nn.functional.batch_norm(
            out_b_57,
            l_self_modules_backbone_modules_layer3_modules_12_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_b_57 = l_self_modules_backbone_modules_layer3_modules_12_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_12_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_12_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_12_modules_bn2_parameters_bias_ = (None)
        out_b_59 = torch.nn.functional.relu(out_b_58, inplace=True)
        out_b_58 = None
        input_219 = torch._C._nn.avg_pool2d(out_b_59, 4, 4, 0, False, True, None)
        input_220 = torch.conv2d(
            input_219,
            l_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k2_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_219 = l_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k2_modules_1_parameters_weight_ = (None)
        input_221 = torch.nn.functional.batch_norm(
            input_220,
            l_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k2_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k2_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k2_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k2_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_220 = l_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k2_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k2_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k2_modules_2_parameters_bias_ = (None)
        interpolate_19 = torch.nn.functional.interpolate(input_221, (16, 12))
        input_221 = None
        add_19 = torch.add(out_b_59, interpolate_19)
        interpolate_19 = None
        out_114 = torch.sigmoid(add_19)
        add_19 = None
        input_222 = torch.conv2d(
            out_b_59,
            l_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_b_59 = l_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k3_modules_0_parameters_weight_ = (None)
        input_223 = torch.nn.functional.batch_norm(
            input_222,
            l_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_222 = l_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k3_modules_1_parameters_bias_ = (None)
        out_115 = torch.mul(input_223, out_114)
        input_223 = out_114 = None
        input_224 = torch.conv2d(
            out_115,
            l_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k4_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_115 = l_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k4_modules_0_parameters_weight_ = (None)
        input_225 = torch.nn.functional.batch_norm(
            input_224,
            l_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k4_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k4_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k4_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_224 = l_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k4_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k4_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_12_modules_scconv_modules_k4_modules_1_parameters_bias_ = (None)
        input_226 = torch.nn.functional.relu(input_225, inplace=True)
        input_225 = None
        cat_19 = torch.cat([input_218, input_226], dim=1)
        input_218 = input_226 = None
        out_116 = torch.conv2d(
            cat_19,
            l_self_modules_backbone_modules_layer3_modules_12_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_19 = l_self_modules_backbone_modules_layer3_modules_12_modules_conv3_parameters_weight_ = (None)
        out_117 = torch.nn.functional.batch_norm(
            out_116,
            l_self_modules_backbone_modules_layer3_modules_12_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_12_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_116 = l_self_modules_backbone_modules_layer3_modules_12_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_12_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_12_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_12_modules_bn3_parameters_bias_ = (None)
        out_117 += out_113
        out_118 = out_117
        out_117 = out_113 = None
        out_119 = torch.nn.functional.relu(out_118, inplace=True)
        out_118 = None
        out_a_60 = torch.conv2d(
            out_119,
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
        out_a_61 = torch.nn.functional.batch_norm(
            out_a_60,
            l_self_modules_backbone_modules_layer3_modules_13_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_a_60 = l_self_modules_backbone_modules_layer3_modules_13_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_13_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_13_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_13_modules_bn1_parameters_bias_ = (None)
        out_a_62 = torch.nn.functional.relu(out_a_61, inplace=True)
        out_a_61 = None
        input_227 = torch.conv2d(
            out_a_62,
            l_self_modules_backbone_modules_layer3_modules_13_modules_k1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_a_62 = l_self_modules_backbone_modules_layer3_modules_13_modules_k1_modules_0_parameters_weight_ = (None)
        input_228 = torch.nn.functional.batch_norm(
            input_227,
            l_self_modules_backbone_modules_layer3_modules_13_modules_k1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_k1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_k1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_k1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_227 = l_self_modules_backbone_modules_layer3_modules_13_modules_k1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_13_modules_k1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_13_modules_k1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_13_modules_k1_modules_1_parameters_bias_ = (None)
        input_229 = torch.nn.functional.relu(input_228, inplace=True)
        input_228 = None
        out_b_60 = torch.conv2d(
            out_119,
            l_self_modules_backbone_modules_layer3_modules_13_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_13_modules_conv2_parameters_weight_ = (
            None
        )
        out_b_61 = torch.nn.functional.batch_norm(
            out_b_60,
            l_self_modules_backbone_modules_layer3_modules_13_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_b_60 = l_self_modules_backbone_modules_layer3_modules_13_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_13_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_13_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_13_modules_bn2_parameters_bias_ = (None)
        out_b_62 = torch.nn.functional.relu(out_b_61, inplace=True)
        out_b_61 = None
        input_230 = torch._C._nn.avg_pool2d(out_b_62, 4, 4, 0, False, True, None)
        input_231 = torch.conv2d(
            input_230,
            l_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k2_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_230 = l_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k2_modules_1_parameters_weight_ = (None)
        input_232 = torch.nn.functional.batch_norm(
            input_231,
            l_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k2_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k2_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k2_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k2_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_231 = l_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k2_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k2_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k2_modules_2_parameters_bias_ = (None)
        interpolate_20 = torch.nn.functional.interpolate(input_232, (16, 12))
        input_232 = None
        add_20 = torch.add(out_b_62, interpolate_20)
        interpolate_20 = None
        out_120 = torch.sigmoid(add_20)
        add_20 = None
        input_233 = torch.conv2d(
            out_b_62,
            l_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_b_62 = l_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k3_modules_0_parameters_weight_ = (None)
        input_234 = torch.nn.functional.batch_norm(
            input_233,
            l_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_233 = l_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k3_modules_1_parameters_bias_ = (None)
        out_121 = torch.mul(input_234, out_120)
        input_234 = out_120 = None
        input_235 = torch.conv2d(
            out_121,
            l_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k4_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_121 = l_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k4_modules_0_parameters_weight_ = (None)
        input_236 = torch.nn.functional.batch_norm(
            input_235,
            l_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k4_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k4_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k4_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_235 = l_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k4_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k4_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_13_modules_scconv_modules_k4_modules_1_parameters_bias_ = (None)
        input_237 = torch.nn.functional.relu(input_236, inplace=True)
        input_236 = None
        cat_20 = torch.cat([input_229, input_237], dim=1)
        input_229 = input_237 = None
        out_122 = torch.conv2d(
            cat_20,
            l_self_modules_backbone_modules_layer3_modules_13_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_20 = l_self_modules_backbone_modules_layer3_modules_13_modules_conv3_parameters_weight_ = (None)
        out_123 = torch.nn.functional.batch_norm(
            out_122,
            l_self_modules_backbone_modules_layer3_modules_13_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_13_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_122 = l_self_modules_backbone_modules_layer3_modules_13_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_13_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_13_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_13_modules_bn3_parameters_bias_ = (None)
        out_123 += out_119
        out_124 = out_123
        out_123 = out_119 = None
        out_125 = torch.nn.functional.relu(out_124, inplace=True)
        out_124 = None
        out_a_63 = torch.conv2d(
            out_125,
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
        out_a_64 = torch.nn.functional.batch_norm(
            out_a_63,
            l_self_modules_backbone_modules_layer3_modules_14_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_a_63 = l_self_modules_backbone_modules_layer3_modules_14_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_14_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_14_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_14_modules_bn1_parameters_bias_ = (None)
        out_a_65 = torch.nn.functional.relu(out_a_64, inplace=True)
        out_a_64 = None
        input_238 = torch.conv2d(
            out_a_65,
            l_self_modules_backbone_modules_layer3_modules_14_modules_k1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_a_65 = l_self_modules_backbone_modules_layer3_modules_14_modules_k1_modules_0_parameters_weight_ = (None)
        input_239 = torch.nn.functional.batch_norm(
            input_238,
            l_self_modules_backbone_modules_layer3_modules_14_modules_k1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_k1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_k1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_k1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_238 = l_self_modules_backbone_modules_layer3_modules_14_modules_k1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_14_modules_k1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_14_modules_k1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_14_modules_k1_modules_1_parameters_bias_ = (None)
        input_240 = torch.nn.functional.relu(input_239, inplace=True)
        input_239 = None
        out_b_63 = torch.conv2d(
            out_125,
            l_self_modules_backbone_modules_layer3_modules_14_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_14_modules_conv2_parameters_weight_ = (
            None
        )
        out_b_64 = torch.nn.functional.batch_norm(
            out_b_63,
            l_self_modules_backbone_modules_layer3_modules_14_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_b_63 = l_self_modules_backbone_modules_layer3_modules_14_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_14_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_14_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_14_modules_bn2_parameters_bias_ = (None)
        out_b_65 = torch.nn.functional.relu(out_b_64, inplace=True)
        out_b_64 = None
        input_241 = torch._C._nn.avg_pool2d(out_b_65, 4, 4, 0, False, True, None)
        input_242 = torch.conv2d(
            input_241,
            l_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k2_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_241 = l_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k2_modules_1_parameters_weight_ = (None)
        input_243 = torch.nn.functional.batch_norm(
            input_242,
            l_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k2_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k2_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k2_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k2_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_242 = l_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k2_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k2_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k2_modules_2_parameters_bias_ = (None)
        interpolate_21 = torch.nn.functional.interpolate(input_243, (16, 12))
        input_243 = None
        add_21 = torch.add(out_b_65, interpolate_21)
        interpolate_21 = None
        out_126 = torch.sigmoid(add_21)
        add_21 = None
        input_244 = torch.conv2d(
            out_b_65,
            l_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_b_65 = l_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k3_modules_0_parameters_weight_ = (None)
        input_245 = torch.nn.functional.batch_norm(
            input_244,
            l_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_244 = l_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k3_modules_1_parameters_bias_ = (None)
        out_127 = torch.mul(input_245, out_126)
        input_245 = out_126 = None
        input_246 = torch.conv2d(
            out_127,
            l_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k4_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_127 = l_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k4_modules_0_parameters_weight_ = (None)
        input_247 = torch.nn.functional.batch_norm(
            input_246,
            l_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k4_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k4_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k4_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_246 = l_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k4_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k4_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_14_modules_scconv_modules_k4_modules_1_parameters_bias_ = (None)
        input_248 = torch.nn.functional.relu(input_247, inplace=True)
        input_247 = None
        cat_21 = torch.cat([input_240, input_248], dim=1)
        input_240 = input_248 = None
        out_128 = torch.conv2d(
            cat_21,
            l_self_modules_backbone_modules_layer3_modules_14_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_21 = l_self_modules_backbone_modules_layer3_modules_14_modules_conv3_parameters_weight_ = (None)
        out_129 = torch.nn.functional.batch_norm(
            out_128,
            l_self_modules_backbone_modules_layer3_modules_14_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_14_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_128 = l_self_modules_backbone_modules_layer3_modules_14_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_14_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_14_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_14_modules_bn3_parameters_bias_ = (None)
        out_129 += out_125
        out_130 = out_129
        out_129 = out_125 = None
        out_131 = torch.nn.functional.relu(out_130, inplace=True)
        out_130 = None
        out_a_66 = torch.conv2d(
            out_131,
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
        out_a_67 = torch.nn.functional.batch_norm(
            out_a_66,
            l_self_modules_backbone_modules_layer3_modules_15_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_a_66 = l_self_modules_backbone_modules_layer3_modules_15_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_15_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_15_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_15_modules_bn1_parameters_bias_ = (None)
        out_a_68 = torch.nn.functional.relu(out_a_67, inplace=True)
        out_a_67 = None
        input_249 = torch.conv2d(
            out_a_68,
            l_self_modules_backbone_modules_layer3_modules_15_modules_k1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_a_68 = l_self_modules_backbone_modules_layer3_modules_15_modules_k1_modules_0_parameters_weight_ = (None)
        input_250 = torch.nn.functional.batch_norm(
            input_249,
            l_self_modules_backbone_modules_layer3_modules_15_modules_k1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_k1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_k1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_k1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_249 = l_self_modules_backbone_modules_layer3_modules_15_modules_k1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_15_modules_k1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_15_modules_k1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_15_modules_k1_modules_1_parameters_bias_ = (None)
        input_251 = torch.nn.functional.relu(input_250, inplace=True)
        input_250 = None
        out_b_66 = torch.conv2d(
            out_131,
            l_self_modules_backbone_modules_layer3_modules_15_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_15_modules_conv2_parameters_weight_ = (
            None
        )
        out_b_67 = torch.nn.functional.batch_norm(
            out_b_66,
            l_self_modules_backbone_modules_layer3_modules_15_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_b_66 = l_self_modules_backbone_modules_layer3_modules_15_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_15_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_15_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_15_modules_bn2_parameters_bias_ = (None)
        out_b_68 = torch.nn.functional.relu(out_b_67, inplace=True)
        out_b_67 = None
        input_252 = torch._C._nn.avg_pool2d(out_b_68, 4, 4, 0, False, True, None)
        input_253 = torch.conv2d(
            input_252,
            l_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k2_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_252 = l_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k2_modules_1_parameters_weight_ = (None)
        input_254 = torch.nn.functional.batch_norm(
            input_253,
            l_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k2_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k2_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k2_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k2_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_253 = l_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k2_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k2_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k2_modules_2_parameters_bias_ = (None)
        interpolate_22 = torch.nn.functional.interpolate(input_254, (16, 12))
        input_254 = None
        add_22 = torch.add(out_b_68, interpolate_22)
        interpolate_22 = None
        out_132 = torch.sigmoid(add_22)
        add_22 = None
        input_255 = torch.conv2d(
            out_b_68,
            l_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_b_68 = l_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k3_modules_0_parameters_weight_ = (None)
        input_256 = torch.nn.functional.batch_norm(
            input_255,
            l_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_255 = l_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k3_modules_1_parameters_bias_ = (None)
        out_133 = torch.mul(input_256, out_132)
        input_256 = out_132 = None
        input_257 = torch.conv2d(
            out_133,
            l_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k4_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_133 = l_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k4_modules_0_parameters_weight_ = (None)
        input_258 = torch.nn.functional.batch_norm(
            input_257,
            l_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k4_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k4_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k4_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_257 = l_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k4_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k4_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_15_modules_scconv_modules_k4_modules_1_parameters_bias_ = (None)
        input_259 = torch.nn.functional.relu(input_258, inplace=True)
        input_258 = None
        cat_22 = torch.cat([input_251, input_259], dim=1)
        input_251 = input_259 = None
        out_134 = torch.conv2d(
            cat_22,
            l_self_modules_backbone_modules_layer3_modules_15_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_22 = l_self_modules_backbone_modules_layer3_modules_15_modules_conv3_parameters_weight_ = (None)
        out_135 = torch.nn.functional.batch_norm(
            out_134,
            l_self_modules_backbone_modules_layer3_modules_15_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_15_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_134 = l_self_modules_backbone_modules_layer3_modules_15_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_15_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_15_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_15_modules_bn3_parameters_bias_ = (None)
        out_135 += out_131
        out_136 = out_135
        out_135 = out_131 = None
        out_137 = torch.nn.functional.relu(out_136, inplace=True)
        out_136 = None
        out_a_69 = torch.conv2d(
            out_137,
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
        out_a_70 = torch.nn.functional.batch_norm(
            out_a_69,
            l_self_modules_backbone_modules_layer3_modules_16_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_a_69 = l_self_modules_backbone_modules_layer3_modules_16_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_16_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_16_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_16_modules_bn1_parameters_bias_ = (None)
        out_a_71 = torch.nn.functional.relu(out_a_70, inplace=True)
        out_a_70 = None
        input_260 = torch.conv2d(
            out_a_71,
            l_self_modules_backbone_modules_layer3_modules_16_modules_k1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_a_71 = l_self_modules_backbone_modules_layer3_modules_16_modules_k1_modules_0_parameters_weight_ = (None)
        input_261 = torch.nn.functional.batch_norm(
            input_260,
            l_self_modules_backbone_modules_layer3_modules_16_modules_k1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_k1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_k1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_k1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_260 = l_self_modules_backbone_modules_layer3_modules_16_modules_k1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_16_modules_k1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_16_modules_k1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_16_modules_k1_modules_1_parameters_bias_ = (None)
        input_262 = torch.nn.functional.relu(input_261, inplace=True)
        input_261 = None
        out_b_69 = torch.conv2d(
            out_137,
            l_self_modules_backbone_modules_layer3_modules_16_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_16_modules_conv2_parameters_weight_ = (
            None
        )
        out_b_70 = torch.nn.functional.batch_norm(
            out_b_69,
            l_self_modules_backbone_modules_layer3_modules_16_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_b_69 = l_self_modules_backbone_modules_layer3_modules_16_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_16_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_16_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_16_modules_bn2_parameters_bias_ = (None)
        out_b_71 = torch.nn.functional.relu(out_b_70, inplace=True)
        out_b_70 = None
        input_263 = torch._C._nn.avg_pool2d(out_b_71, 4, 4, 0, False, True, None)
        input_264 = torch.conv2d(
            input_263,
            l_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k2_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_263 = l_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k2_modules_1_parameters_weight_ = (None)
        input_265 = torch.nn.functional.batch_norm(
            input_264,
            l_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k2_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k2_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k2_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k2_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_264 = l_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k2_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k2_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k2_modules_2_parameters_bias_ = (None)
        interpolate_23 = torch.nn.functional.interpolate(input_265, (16, 12))
        input_265 = None
        add_23 = torch.add(out_b_71, interpolate_23)
        interpolate_23 = None
        out_138 = torch.sigmoid(add_23)
        add_23 = None
        input_266 = torch.conv2d(
            out_b_71,
            l_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_b_71 = l_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k3_modules_0_parameters_weight_ = (None)
        input_267 = torch.nn.functional.batch_norm(
            input_266,
            l_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_266 = l_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k3_modules_1_parameters_bias_ = (None)
        out_139 = torch.mul(input_267, out_138)
        input_267 = out_138 = None
        input_268 = torch.conv2d(
            out_139,
            l_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k4_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_139 = l_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k4_modules_0_parameters_weight_ = (None)
        input_269 = torch.nn.functional.batch_norm(
            input_268,
            l_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k4_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k4_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k4_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_268 = l_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k4_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k4_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_16_modules_scconv_modules_k4_modules_1_parameters_bias_ = (None)
        input_270 = torch.nn.functional.relu(input_269, inplace=True)
        input_269 = None
        cat_23 = torch.cat([input_262, input_270], dim=1)
        input_262 = input_270 = None
        out_140 = torch.conv2d(
            cat_23,
            l_self_modules_backbone_modules_layer3_modules_16_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_23 = l_self_modules_backbone_modules_layer3_modules_16_modules_conv3_parameters_weight_ = (None)
        out_141 = torch.nn.functional.batch_norm(
            out_140,
            l_self_modules_backbone_modules_layer3_modules_16_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_16_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_140 = l_self_modules_backbone_modules_layer3_modules_16_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_16_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_16_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_16_modules_bn3_parameters_bias_ = (None)
        out_141 += out_137
        out_142 = out_141
        out_141 = out_137 = None
        out_143 = torch.nn.functional.relu(out_142, inplace=True)
        out_142 = None
        out_a_72 = torch.conv2d(
            out_143,
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
        out_a_73 = torch.nn.functional.batch_norm(
            out_a_72,
            l_self_modules_backbone_modules_layer3_modules_17_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_a_72 = l_self_modules_backbone_modules_layer3_modules_17_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_17_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_17_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_17_modules_bn1_parameters_bias_ = (None)
        out_a_74 = torch.nn.functional.relu(out_a_73, inplace=True)
        out_a_73 = None
        input_271 = torch.conv2d(
            out_a_74,
            l_self_modules_backbone_modules_layer3_modules_17_modules_k1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_a_74 = l_self_modules_backbone_modules_layer3_modules_17_modules_k1_modules_0_parameters_weight_ = (None)
        input_272 = torch.nn.functional.batch_norm(
            input_271,
            l_self_modules_backbone_modules_layer3_modules_17_modules_k1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_k1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_k1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_k1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_271 = l_self_modules_backbone_modules_layer3_modules_17_modules_k1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_17_modules_k1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_17_modules_k1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_17_modules_k1_modules_1_parameters_bias_ = (None)
        input_273 = torch.nn.functional.relu(input_272, inplace=True)
        input_272 = None
        out_b_72 = torch.conv2d(
            out_143,
            l_self_modules_backbone_modules_layer3_modules_17_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_17_modules_conv2_parameters_weight_ = (
            None
        )
        out_b_73 = torch.nn.functional.batch_norm(
            out_b_72,
            l_self_modules_backbone_modules_layer3_modules_17_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_b_72 = l_self_modules_backbone_modules_layer3_modules_17_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_17_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_17_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_17_modules_bn2_parameters_bias_ = (None)
        out_b_74 = torch.nn.functional.relu(out_b_73, inplace=True)
        out_b_73 = None
        input_274 = torch._C._nn.avg_pool2d(out_b_74, 4, 4, 0, False, True, None)
        input_275 = torch.conv2d(
            input_274,
            l_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k2_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_274 = l_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k2_modules_1_parameters_weight_ = (None)
        input_276 = torch.nn.functional.batch_norm(
            input_275,
            l_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k2_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k2_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k2_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k2_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_275 = l_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k2_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k2_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k2_modules_2_parameters_bias_ = (None)
        interpolate_24 = torch.nn.functional.interpolate(input_276, (16, 12))
        input_276 = None
        add_24 = torch.add(out_b_74, interpolate_24)
        interpolate_24 = None
        out_144 = torch.sigmoid(add_24)
        add_24 = None
        input_277 = torch.conv2d(
            out_b_74,
            l_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_b_74 = l_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k3_modules_0_parameters_weight_ = (None)
        input_278 = torch.nn.functional.batch_norm(
            input_277,
            l_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_277 = l_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k3_modules_1_parameters_bias_ = (None)
        out_145 = torch.mul(input_278, out_144)
        input_278 = out_144 = None
        input_279 = torch.conv2d(
            out_145,
            l_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k4_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_145 = l_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k4_modules_0_parameters_weight_ = (None)
        input_280 = torch.nn.functional.batch_norm(
            input_279,
            l_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k4_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k4_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k4_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_279 = l_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k4_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k4_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_17_modules_scconv_modules_k4_modules_1_parameters_bias_ = (None)
        input_281 = torch.nn.functional.relu(input_280, inplace=True)
        input_280 = None
        cat_24 = torch.cat([input_273, input_281], dim=1)
        input_273 = input_281 = None
        out_146 = torch.conv2d(
            cat_24,
            l_self_modules_backbone_modules_layer3_modules_17_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_24 = l_self_modules_backbone_modules_layer3_modules_17_modules_conv3_parameters_weight_ = (None)
        out_147 = torch.nn.functional.batch_norm(
            out_146,
            l_self_modules_backbone_modules_layer3_modules_17_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_17_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_146 = l_self_modules_backbone_modules_layer3_modules_17_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_17_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_17_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_17_modules_bn3_parameters_bias_ = (None)
        out_147 += out_143
        out_148 = out_147
        out_147 = out_143 = None
        out_149 = torch.nn.functional.relu(out_148, inplace=True)
        out_148 = None
        out_a_75 = torch.conv2d(
            out_149,
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
        out_a_76 = torch.nn.functional.batch_norm(
            out_a_75,
            l_self_modules_backbone_modules_layer3_modules_18_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_a_75 = l_self_modules_backbone_modules_layer3_modules_18_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_18_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_18_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_18_modules_bn1_parameters_bias_ = (None)
        out_a_77 = torch.nn.functional.relu(out_a_76, inplace=True)
        out_a_76 = None
        input_282 = torch.conv2d(
            out_a_77,
            l_self_modules_backbone_modules_layer3_modules_18_modules_k1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_a_77 = l_self_modules_backbone_modules_layer3_modules_18_modules_k1_modules_0_parameters_weight_ = (None)
        input_283 = torch.nn.functional.batch_norm(
            input_282,
            l_self_modules_backbone_modules_layer3_modules_18_modules_k1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_k1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_k1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_k1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_282 = l_self_modules_backbone_modules_layer3_modules_18_modules_k1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_18_modules_k1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_18_modules_k1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_18_modules_k1_modules_1_parameters_bias_ = (None)
        input_284 = torch.nn.functional.relu(input_283, inplace=True)
        input_283 = None
        out_b_75 = torch.conv2d(
            out_149,
            l_self_modules_backbone_modules_layer3_modules_18_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_18_modules_conv2_parameters_weight_ = (
            None
        )
        out_b_76 = torch.nn.functional.batch_norm(
            out_b_75,
            l_self_modules_backbone_modules_layer3_modules_18_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_b_75 = l_self_modules_backbone_modules_layer3_modules_18_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_18_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_18_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_18_modules_bn2_parameters_bias_ = (None)
        out_b_77 = torch.nn.functional.relu(out_b_76, inplace=True)
        out_b_76 = None
        input_285 = torch._C._nn.avg_pool2d(out_b_77, 4, 4, 0, False, True, None)
        input_286 = torch.conv2d(
            input_285,
            l_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k2_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_285 = l_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k2_modules_1_parameters_weight_ = (None)
        input_287 = torch.nn.functional.batch_norm(
            input_286,
            l_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k2_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k2_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k2_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k2_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_286 = l_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k2_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k2_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k2_modules_2_parameters_bias_ = (None)
        interpolate_25 = torch.nn.functional.interpolate(input_287, (16, 12))
        input_287 = None
        add_25 = torch.add(out_b_77, interpolate_25)
        interpolate_25 = None
        out_150 = torch.sigmoid(add_25)
        add_25 = None
        input_288 = torch.conv2d(
            out_b_77,
            l_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_b_77 = l_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k3_modules_0_parameters_weight_ = (None)
        input_289 = torch.nn.functional.batch_norm(
            input_288,
            l_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_288 = l_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k3_modules_1_parameters_bias_ = (None)
        out_151 = torch.mul(input_289, out_150)
        input_289 = out_150 = None
        input_290 = torch.conv2d(
            out_151,
            l_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k4_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_151 = l_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k4_modules_0_parameters_weight_ = (None)
        input_291 = torch.nn.functional.batch_norm(
            input_290,
            l_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k4_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k4_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k4_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_290 = l_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k4_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k4_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_18_modules_scconv_modules_k4_modules_1_parameters_bias_ = (None)
        input_292 = torch.nn.functional.relu(input_291, inplace=True)
        input_291 = None
        cat_25 = torch.cat([input_284, input_292], dim=1)
        input_284 = input_292 = None
        out_152 = torch.conv2d(
            cat_25,
            l_self_modules_backbone_modules_layer3_modules_18_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_25 = l_self_modules_backbone_modules_layer3_modules_18_modules_conv3_parameters_weight_ = (None)
        out_153 = torch.nn.functional.batch_norm(
            out_152,
            l_self_modules_backbone_modules_layer3_modules_18_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_18_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_152 = l_self_modules_backbone_modules_layer3_modules_18_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_18_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_18_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_18_modules_bn3_parameters_bias_ = (None)
        out_153 += out_149
        out_154 = out_153
        out_153 = out_149 = None
        out_155 = torch.nn.functional.relu(out_154, inplace=True)
        out_154 = None
        out_a_78 = torch.conv2d(
            out_155,
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
        out_a_79 = torch.nn.functional.batch_norm(
            out_a_78,
            l_self_modules_backbone_modules_layer3_modules_19_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_a_78 = l_self_modules_backbone_modules_layer3_modules_19_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_19_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_19_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_19_modules_bn1_parameters_bias_ = (None)
        out_a_80 = torch.nn.functional.relu(out_a_79, inplace=True)
        out_a_79 = None
        input_293 = torch.conv2d(
            out_a_80,
            l_self_modules_backbone_modules_layer3_modules_19_modules_k1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_a_80 = l_self_modules_backbone_modules_layer3_modules_19_modules_k1_modules_0_parameters_weight_ = (None)
        input_294 = torch.nn.functional.batch_norm(
            input_293,
            l_self_modules_backbone_modules_layer3_modules_19_modules_k1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_k1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_k1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_k1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_293 = l_self_modules_backbone_modules_layer3_modules_19_modules_k1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_19_modules_k1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_19_modules_k1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_19_modules_k1_modules_1_parameters_bias_ = (None)
        input_295 = torch.nn.functional.relu(input_294, inplace=True)
        input_294 = None
        out_b_78 = torch.conv2d(
            out_155,
            l_self_modules_backbone_modules_layer3_modules_19_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_19_modules_conv2_parameters_weight_ = (
            None
        )
        out_b_79 = torch.nn.functional.batch_norm(
            out_b_78,
            l_self_modules_backbone_modules_layer3_modules_19_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_b_78 = l_self_modules_backbone_modules_layer3_modules_19_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_19_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_19_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_19_modules_bn2_parameters_bias_ = (None)
        out_b_80 = torch.nn.functional.relu(out_b_79, inplace=True)
        out_b_79 = None
        input_296 = torch._C._nn.avg_pool2d(out_b_80, 4, 4, 0, False, True, None)
        input_297 = torch.conv2d(
            input_296,
            l_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k2_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_296 = l_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k2_modules_1_parameters_weight_ = (None)
        input_298 = torch.nn.functional.batch_norm(
            input_297,
            l_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k2_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k2_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k2_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k2_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_297 = l_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k2_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k2_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k2_modules_2_parameters_bias_ = (None)
        interpolate_26 = torch.nn.functional.interpolate(input_298, (16, 12))
        input_298 = None
        add_26 = torch.add(out_b_80, interpolate_26)
        interpolate_26 = None
        out_156 = torch.sigmoid(add_26)
        add_26 = None
        input_299 = torch.conv2d(
            out_b_80,
            l_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_b_80 = l_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k3_modules_0_parameters_weight_ = (None)
        input_300 = torch.nn.functional.batch_norm(
            input_299,
            l_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_299 = l_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k3_modules_1_parameters_bias_ = (None)
        out_157 = torch.mul(input_300, out_156)
        input_300 = out_156 = None
        input_301 = torch.conv2d(
            out_157,
            l_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k4_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_157 = l_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k4_modules_0_parameters_weight_ = (None)
        input_302 = torch.nn.functional.batch_norm(
            input_301,
            l_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k4_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k4_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k4_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_301 = l_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k4_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k4_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_19_modules_scconv_modules_k4_modules_1_parameters_bias_ = (None)
        input_303 = torch.nn.functional.relu(input_302, inplace=True)
        input_302 = None
        cat_26 = torch.cat([input_295, input_303], dim=1)
        input_295 = input_303 = None
        out_158 = torch.conv2d(
            cat_26,
            l_self_modules_backbone_modules_layer3_modules_19_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_26 = l_self_modules_backbone_modules_layer3_modules_19_modules_conv3_parameters_weight_ = (None)
        out_159 = torch.nn.functional.batch_norm(
            out_158,
            l_self_modules_backbone_modules_layer3_modules_19_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_19_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_158 = l_self_modules_backbone_modules_layer3_modules_19_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_19_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_19_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_19_modules_bn3_parameters_bias_ = (None)
        out_159 += out_155
        out_160 = out_159
        out_159 = out_155 = None
        out_161 = torch.nn.functional.relu(out_160, inplace=True)
        out_160 = None
        out_a_81 = torch.conv2d(
            out_161,
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
        out_a_82 = torch.nn.functional.batch_norm(
            out_a_81,
            l_self_modules_backbone_modules_layer3_modules_20_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_a_81 = l_self_modules_backbone_modules_layer3_modules_20_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_20_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_20_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_20_modules_bn1_parameters_bias_ = (None)
        out_a_83 = torch.nn.functional.relu(out_a_82, inplace=True)
        out_a_82 = None
        input_304 = torch.conv2d(
            out_a_83,
            l_self_modules_backbone_modules_layer3_modules_20_modules_k1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_a_83 = l_self_modules_backbone_modules_layer3_modules_20_modules_k1_modules_0_parameters_weight_ = (None)
        input_305 = torch.nn.functional.batch_norm(
            input_304,
            l_self_modules_backbone_modules_layer3_modules_20_modules_k1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_k1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_k1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_k1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_304 = l_self_modules_backbone_modules_layer3_modules_20_modules_k1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_20_modules_k1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_20_modules_k1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_20_modules_k1_modules_1_parameters_bias_ = (None)
        input_306 = torch.nn.functional.relu(input_305, inplace=True)
        input_305 = None
        out_b_81 = torch.conv2d(
            out_161,
            l_self_modules_backbone_modules_layer3_modules_20_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_20_modules_conv2_parameters_weight_ = (
            None
        )
        out_b_82 = torch.nn.functional.batch_norm(
            out_b_81,
            l_self_modules_backbone_modules_layer3_modules_20_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_b_81 = l_self_modules_backbone_modules_layer3_modules_20_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_20_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_20_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_20_modules_bn2_parameters_bias_ = (None)
        out_b_83 = torch.nn.functional.relu(out_b_82, inplace=True)
        out_b_82 = None
        input_307 = torch._C._nn.avg_pool2d(out_b_83, 4, 4, 0, False, True, None)
        input_308 = torch.conv2d(
            input_307,
            l_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k2_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_307 = l_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k2_modules_1_parameters_weight_ = (None)
        input_309 = torch.nn.functional.batch_norm(
            input_308,
            l_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k2_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k2_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k2_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k2_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_308 = l_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k2_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k2_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k2_modules_2_parameters_bias_ = (None)
        interpolate_27 = torch.nn.functional.interpolate(input_309, (16, 12))
        input_309 = None
        add_27 = torch.add(out_b_83, interpolate_27)
        interpolate_27 = None
        out_162 = torch.sigmoid(add_27)
        add_27 = None
        input_310 = torch.conv2d(
            out_b_83,
            l_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_b_83 = l_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k3_modules_0_parameters_weight_ = (None)
        input_311 = torch.nn.functional.batch_norm(
            input_310,
            l_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_310 = l_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k3_modules_1_parameters_bias_ = (None)
        out_163 = torch.mul(input_311, out_162)
        input_311 = out_162 = None
        input_312 = torch.conv2d(
            out_163,
            l_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k4_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_163 = l_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k4_modules_0_parameters_weight_ = (None)
        input_313 = torch.nn.functional.batch_norm(
            input_312,
            l_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k4_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k4_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k4_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_312 = l_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k4_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k4_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_20_modules_scconv_modules_k4_modules_1_parameters_bias_ = (None)
        input_314 = torch.nn.functional.relu(input_313, inplace=True)
        input_313 = None
        cat_27 = torch.cat([input_306, input_314], dim=1)
        input_306 = input_314 = None
        out_164 = torch.conv2d(
            cat_27,
            l_self_modules_backbone_modules_layer3_modules_20_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_27 = l_self_modules_backbone_modules_layer3_modules_20_modules_conv3_parameters_weight_ = (None)
        out_165 = torch.nn.functional.batch_norm(
            out_164,
            l_self_modules_backbone_modules_layer3_modules_20_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_20_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_164 = l_self_modules_backbone_modules_layer3_modules_20_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_20_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_20_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_20_modules_bn3_parameters_bias_ = (None)
        out_165 += out_161
        out_166 = out_165
        out_165 = out_161 = None
        out_167 = torch.nn.functional.relu(out_166, inplace=True)
        out_166 = None
        out_a_84 = torch.conv2d(
            out_167,
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
        out_a_85 = torch.nn.functional.batch_norm(
            out_a_84,
            l_self_modules_backbone_modules_layer3_modules_21_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_a_84 = l_self_modules_backbone_modules_layer3_modules_21_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_21_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_21_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_21_modules_bn1_parameters_bias_ = (None)
        out_a_86 = torch.nn.functional.relu(out_a_85, inplace=True)
        out_a_85 = None
        input_315 = torch.conv2d(
            out_a_86,
            l_self_modules_backbone_modules_layer3_modules_21_modules_k1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_a_86 = l_self_modules_backbone_modules_layer3_modules_21_modules_k1_modules_0_parameters_weight_ = (None)
        input_316 = torch.nn.functional.batch_norm(
            input_315,
            l_self_modules_backbone_modules_layer3_modules_21_modules_k1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_k1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_k1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_k1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_315 = l_self_modules_backbone_modules_layer3_modules_21_modules_k1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_21_modules_k1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_21_modules_k1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_21_modules_k1_modules_1_parameters_bias_ = (None)
        input_317 = torch.nn.functional.relu(input_316, inplace=True)
        input_316 = None
        out_b_84 = torch.conv2d(
            out_167,
            l_self_modules_backbone_modules_layer3_modules_21_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_21_modules_conv2_parameters_weight_ = (
            None
        )
        out_b_85 = torch.nn.functional.batch_norm(
            out_b_84,
            l_self_modules_backbone_modules_layer3_modules_21_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_b_84 = l_self_modules_backbone_modules_layer3_modules_21_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_21_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_21_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_21_modules_bn2_parameters_bias_ = (None)
        out_b_86 = torch.nn.functional.relu(out_b_85, inplace=True)
        out_b_85 = None
        input_318 = torch._C._nn.avg_pool2d(out_b_86, 4, 4, 0, False, True, None)
        input_319 = torch.conv2d(
            input_318,
            l_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k2_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_318 = l_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k2_modules_1_parameters_weight_ = (None)
        input_320 = torch.nn.functional.batch_norm(
            input_319,
            l_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k2_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k2_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k2_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k2_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_319 = l_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k2_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k2_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k2_modules_2_parameters_bias_ = (None)
        interpolate_28 = torch.nn.functional.interpolate(input_320, (16, 12))
        input_320 = None
        add_28 = torch.add(out_b_86, interpolate_28)
        interpolate_28 = None
        out_168 = torch.sigmoid(add_28)
        add_28 = None
        input_321 = torch.conv2d(
            out_b_86,
            l_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_b_86 = l_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k3_modules_0_parameters_weight_ = (None)
        input_322 = torch.nn.functional.batch_norm(
            input_321,
            l_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_321 = l_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k3_modules_1_parameters_bias_ = (None)
        out_169 = torch.mul(input_322, out_168)
        input_322 = out_168 = None
        input_323 = torch.conv2d(
            out_169,
            l_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k4_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_169 = l_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k4_modules_0_parameters_weight_ = (None)
        input_324 = torch.nn.functional.batch_norm(
            input_323,
            l_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k4_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k4_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k4_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_323 = l_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k4_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k4_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_21_modules_scconv_modules_k4_modules_1_parameters_bias_ = (None)
        input_325 = torch.nn.functional.relu(input_324, inplace=True)
        input_324 = None
        cat_28 = torch.cat([input_317, input_325], dim=1)
        input_317 = input_325 = None
        out_170 = torch.conv2d(
            cat_28,
            l_self_modules_backbone_modules_layer3_modules_21_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_28 = l_self_modules_backbone_modules_layer3_modules_21_modules_conv3_parameters_weight_ = (None)
        out_171 = torch.nn.functional.batch_norm(
            out_170,
            l_self_modules_backbone_modules_layer3_modules_21_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_21_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_170 = l_self_modules_backbone_modules_layer3_modules_21_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_21_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_21_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_21_modules_bn3_parameters_bias_ = (None)
        out_171 += out_167
        out_172 = out_171
        out_171 = out_167 = None
        out_173 = torch.nn.functional.relu(out_172, inplace=True)
        out_172 = None
        out_a_87 = torch.conv2d(
            out_173,
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
        out_a_88 = torch.nn.functional.batch_norm(
            out_a_87,
            l_self_modules_backbone_modules_layer3_modules_22_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_a_87 = l_self_modules_backbone_modules_layer3_modules_22_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_22_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_22_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_22_modules_bn1_parameters_bias_ = (None)
        out_a_89 = torch.nn.functional.relu(out_a_88, inplace=True)
        out_a_88 = None
        input_326 = torch.conv2d(
            out_a_89,
            l_self_modules_backbone_modules_layer3_modules_22_modules_k1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_a_89 = l_self_modules_backbone_modules_layer3_modules_22_modules_k1_modules_0_parameters_weight_ = (None)
        input_327 = torch.nn.functional.batch_norm(
            input_326,
            l_self_modules_backbone_modules_layer3_modules_22_modules_k1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_k1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_k1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_k1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_326 = l_self_modules_backbone_modules_layer3_modules_22_modules_k1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_22_modules_k1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_22_modules_k1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_22_modules_k1_modules_1_parameters_bias_ = (None)
        input_328 = torch.nn.functional.relu(input_327, inplace=True)
        input_327 = None
        out_b_87 = torch.conv2d(
            out_173,
            l_self_modules_backbone_modules_layer3_modules_22_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_22_modules_conv2_parameters_weight_ = (
            None
        )
        out_b_88 = torch.nn.functional.batch_norm(
            out_b_87,
            l_self_modules_backbone_modules_layer3_modules_22_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_b_87 = l_self_modules_backbone_modules_layer3_modules_22_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_22_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_22_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_22_modules_bn2_parameters_bias_ = (None)
        out_b_89 = torch.nn.functional.relu(out_b_88, inplace=True)
        out_b_88 = None
        input_329 = torch._C._nn.avg_pool2d(out_b_89, 4, 4, 0, False, True, None)
        input_330 = torch.conv2d(
            input_329,
            l_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k2_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_329 = l_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k2_modules_1_parameters_weight_ = (None)
        input_331 = torch.nn.functional.batch_norm(
            input_330,
            l_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k2_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k2_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k2_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k2_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_330 = l_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k2_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k2_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k2_modules_2_parameters_bias_ = (None)
        interpolate_29 = torch.nn.functional.interpolate(input_331, (16, 12))
        input_331 = None
        add_29 = torch.add(out_b_89, interpolate_29)
        interpolate_29 = None
        out_174 = torch.sigmoid(add_29)
        add_29 = None
        input_332 = torch.conv2d(
            out_b_89,
            l_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_b_89 = l_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k3_modules_0_parameters_weight_ = (None)
        input_333 = torch.nn.functional.batch_norm(
            input_332,
            l_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_332 = l_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k3_modules_1_parameters_bias_ = (None)
        out_175 = torch.mul(input_333, out_174)
        input_333 = out_174 = None
        input_334 = torch.conv2d(
            out_175,
            l_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k4_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_175 = l_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k4_modules_0_parameters_weight_ = (None)
        input_335 = torch.nn.functional.batch_norm(
            input_334,
            l_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k4_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k4_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k4_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_334 = l_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k4_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k4_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_22_modules_scconv_modules_k4_modules_1_parameters_bias_ = (None)
        input_336 = torch.nn.functional.relu(input_335, inplace=True)
        input_335 = None
        cat_29 = torch.cat([input_328, input_336], dim=1)
        input_328 = input_336 = None
        out_176 = torch.conv2d(
            cat_29,
            l_self_modules_backbone_modules_layer3_modules_22_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_29 = l_self_modules_backbone_modules_layer3_modules_22_modules_conv3_parameters_weight_ = (None)
        out_177 = torch.nn.functional.batch_norm(
            out_176,
            l_self_modules_backbone_modules_layer3_modules_22_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_22_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_176 = l_self_modules_backbone_modules_layer3_modules_22_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_22_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_22_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_22_modules_bn3_parameters_bias_ = (None)
        out_177 += out_173
        out_178 = out_177
        out_177 = out_173 = None
        out_179 = torch.nn.functional.relu(out_178, inplace=True)
        out_178 = None
        out_a_90 = torch.conv2d(
            out_179,
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
        out_a_91 = torch.nn.functional.batch_norm(
            out_a_90,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_a_90 = l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_bias_ = (None)
        out_a_92 = torch.nn.functional.relu(out_a_91, inplace=True)
        out_a_91 = None
        input_337 = torch.conv2d(
            out_a_92,
            l_self_modules_backbone_modules_layer4_modules_0_modules_k1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        out_a_92 = l_self_modules_backbone_modules_layer4_modules_0_modules_k1_modules_0_parameters_weight_ = (None)
        input_338 = torch.nn.functional.batch_norm(
            input_337,
            l_self_modules_backbone_modules_layer4_modules_0_modules_k1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_k1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_k1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_k1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_337 = l_self_modules_backbone_modules_layer4_modules_0_modules_k1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_0_modules_k1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_0_modules_k1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_k1_modules_1_parameters_bias_ = (None)
        input_339 = torch.nn.functional.relu(input_338, inplace=True)
        input_338 = None
        out_b_90 = torch.conv2d(
            out_179,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_parameters_weight_ = (
            None
        )
        out_b_91 = torch.nn.functional.batch_norm(
            out_b_90,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_b_90 = l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_bias_ = (None)
        out_b_92 = torch.nn.functional.relu(out_b_91, inplace=True)
        out_b_91 = None
        input_340 = torch._C._nn.avg_pool2d(out_b_92, 4, 4, 0, False, True, None)
        input_341 = torch.conv2d(
            input_340,
            l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k2_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_340 = l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k2_modules_1_parameters_weight_ = (None)
        input_342 = torch.nn.functional.batch_norm(
            input_341,
            l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k2_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k2_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_341 = l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k2_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k2_modules_2_parameters_bias_ = (None)
        interpolate_30 = torch.nn.functional.interpolate(input_342, (16, 12))
        input_342 = None
        add_30 = torch.add(out_b_92, interpolate_30)
        interpolate_30 = None
        out_180 = torch.sigmoid(add_30)
        add_30 = None
        input_343 = torch.conv2d(
            out_b_92,
            l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_b_92 = l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k3_modules_0_parameters_weight_ = (None)
        input_344 = torch.nn.functional.batch_norm(
            input_343,
            l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_343 = l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k3_modules_1_parameters_bias_ = (None)
        out_181 = torch.mul(input_344, out_180)
        input_344 = out_180 = None
        input_345 = torch.conv2d(
            out_181,
            l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k4_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        out_181 = l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k4_modules_0_parameters_weight_ = (None)
        input_346 = torch.nn.functional.batch_norm(
            input_345,
            l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k4_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_345 = l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k4_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k4_modules_1_parameters_bias_ = (None)
        input_347 = torch.nn.functional.relu(input_346, inplace=True)
        input_346 = None
        cat_30 = torch.cat([input_339, input_347], dim=1)
        input_339 = input_347 = None
        out_182 = torch.conv2d(
            cat_30,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_30 = l_self_modules_backbone_modules_layer4_modules_0_modules_conv3_parameters_weight_ = (None)
        out_183 = torch.nn.functional.batch_norm(
            out_182,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_182 = l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_bias_ = (None)
        input_348 = torch.conv2d(
            out_179,
            l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        out_179 = l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_349 = torch.nn.functional.batch_norm(
            input_348,
            l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_348 = l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        out_183 += input_349
        out_184 = out_183
        out_183 = input_349 = None
        out_185 = torch.nn.functional.relu(out_184, inplace=True)
        out_184 = None
        out_a_93 = torch.conv2d(
            out_185,
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
        out_a_94 = torch.nn.functional.batch_norm(
            out_a_93,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_a_93 = l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_bias_ = (None)
        out_a_95 = torch.nn.functional.relu(out_a_94, inplace=True)
        out_a_94 = None
        input_350 = torch.conv2d(
            out_a_95,
            l_self_modules_backbone_modules_layer4_modules_1_modules_k1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_a_95 = l_self_modules_backbone_modules_layer4_modules_1_modules_k1_modules_0_parameters_weight_ = (None)
        input_351 = torch.nn.functional.batch_norm(
            input_350,
            l_self_modules_backbone_modules_layer4_modules_1_modules_k1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_k1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_k1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_k1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_350 = l_self_modules_backbone_modules_layer4_modules_1_modules_k1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_1_modules_k1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_1_modules_k1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_1_modules_k1_modules_1_parameters_bias_ = (None)
        input_352 = torch.nn.functional.relu(input_351, inplace=True)
        input_351 = None
        out_b_93 = torch.conv2d(
            out_185,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_parameters_weight_ = (
            None
        )
        out_b_94 = torch.nn.functional.batch_norm(
            out_b_93,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_b_93 = l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_bias_ = (None)
        out_b_95 = torch.nn.functional.relu(out_b_94, inplace=True)
        out_b_94 = None
        input_353 = torch._C._nn.avg_pool2d(out_b_95, 4, 4, 0, False, True, None)
        input_354 = torch.conv2d(
            input_353,
            l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k2_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_353 = l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k2_modules_1_parameters_weight_ = (None)
        input_355 = torch.nn.functional.batch_norm(
            input_354,
            l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k2_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k2_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_354 = l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k2_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k2_modules_2_parameters_bias_ = (None)
        interpolate_31 = torch.nn.functional.interpolate(input_355, (8, 6))
        input_355 = None
        add_31 = torch.add(out_b_95, interpolate_31)
        interpolate_31 = None
        out_186 = torch.sigmoid(add_31)
        add_31 = None
        input_356 = torch.conv2d(
            out_b_95,
            l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_b_95 = l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k3_modules_0_parameters_weight_ = (None)
        input_357 = torch.nn.functional.batch_norm(
            input_356,
            l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_356 = l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k3_modules_1_parameters_bias_ = (None)
        out_187 = torch.mul(input_357, out_186)
        input_357 = out_186 = None
        input_358 = torch.conv2d(
            out_187,
            l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k4_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_187 = l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k4_modules_0_parameters_weight_ = (None)
        input_359 = torch.nn.functional.batch_norm(
            input_358,
            l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k4_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_358 = l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k4_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k4_modules_1_parameters_bias_ = (None)
        input_360 = torch.nn.functional.relu(input_359, inplace=True)
        input_359 = None
        cat_31 = torch.cat([input_352, input_360], dim=1)
        input_352 = input_360 = None
        out_188 = torch.conv2d(
            cat_31,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_31 = l_self_modules_backbone_modules_layer4_modules_1_modules_conv3_parameters_weight_ = (None)
        out_189 = torch.nn.functional.batch_norm(
            out_188,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_188 = l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_parameters_bias_ = (None)
        out_189 += out_185
        out_190 = out_189
        out_189 = out_185 = None
        out_191 = torch.nn.functional.relu(out_190, inplace=True)
        out_190 = None
        out_a_96 = torch.conv2d(
            out_191,
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
        out_a_97 = torch.nn.functional.batch_norm(
            out_a_96,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_a_96 = l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_parameters_bias_ = (None)
        out_a_98 = torch.nn.functional.relu(out_a_97, inplace=True)
        out_a_97 = None
        input_361 = torch.conv2d(
            out_a_98,
            l_self_modules_backbone_modules_layer4_modules_2_modules_k1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_a_98 = l_self_modules_backbone_modules_layer4_modules_2_modules_k1_modules_0_parameters_weight_ = (None)
        input_362 = torch.nn.functional.batch_norm(
            input_361,
            l_self_modules_backbone_modules_layer4_modules_2_modules_k1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_k1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_k1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_k1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_361 = l_self_modules_backbone_modules_layer4_modules_2_modules_k1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_2_modules_k1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_2_modules_k1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_2_modules_k1_modules_1_parameters_bias_ = (None)
        input_363 = torch.nn.functional.relu(input_362, inplace=True)
        input_362 = None
        out_b_96 = torch.conv2d(
            out_191,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_parameters_weight_ = (
            None
        )
        out_b_97 = torch.nn.functional.batch_norm(
            out_b_96,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_b_96 = l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_parameters_bias_ = (None)
        out_b_98 = torch.nn.functional.relu(out_b_97, inplace=True)
        out_b_97 = None
        input_364 = torch._C._nn.avg_pool2d(out_b_98, 4, 4, 0, False, True, None)
        input_365 = torch.conv2d(
            input_364,
            l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k2_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_364 = l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k2_modules_1_parameters_weight_ = (None)
        input_366 = torch.nn.functional.batch_norm(
            input_365,
            l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k2_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k2_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_365 = l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k2_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k2_modules_2_parameters_bias_ = (None)
        interpolate_32 = torch.nn.functional.interpolate(input_366, (8, 6))
        input_366 = None
        add_32 = torch.add(out_b_98, interpolate_32)
        interpolate_32 = None
        out_192 = torch.sigmoid(add_32)
        add_32 = None
        input_367 = torch.conv2d(
            out_b_98,
            l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_b_98 = l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k3_modules_0_parameters_weight_ = (None)
        input_368 = torch.nn.functional.batch_norm(
            input_367,
            l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_367 = l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k3_modules_1_parameters_bias_ = (None)
        out_193 = torch.mul(input_368, out_192)
        input_368 = out_192 = None
        input_369 = torch.conv2d(
            out_193,
            l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k4_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_193 = l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k4_modules_0_parameters_weight_ = (None)
        input_370 = torch.nn.functional.batch_norm(
            input_369,
            l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k4_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_369 = l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k4_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k4_modules_1_parameters_bias_ = (None)
        input_371 = torch.nn.functional.relu(input_370, inplace=True)
        input_370 = None
        cat_32 = torch.cat([input_363, input_371], dim=1)
        input_363 = input_371 = None
        out_194 = torch.conv2d(
            cat_32,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_32 = l_self_modules_backbone_modules_layer4_modules_2_modules_conv3_parameters_weight_ = (None)
        out_195 = torch.nn.functional.batch_norm(
            out_194,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_194 = l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_parameters_bias_ = (None)
        out_195 += out_191
        out_196 = out_195
        out_195 = out_191 = None
        out_197 = torch.nn.functional.relu(out_196, inplace=True)
        out_196 = None
        input_372 = torch.conv_transpose2d(
            out_197,
            l_self_modules_head_modules_deconv_layers_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (0, 0),
            1,
            (1, 1),
        )
        out_197 = (
            l_self_modules_head_modules_deconv_layers_modules_0_parameters_weight_
        ) = None
        input_373 = torch.nn.functional.batch_norm(
            input_372,
            l_self_modules_head_modules_deconv_layers_modules_1_buffers_running_mean_,
            l_self_modules_head_modules_deconv_layers_modules_1_buffers_running_var_,
            l_self_modules_head_modules_deconv_layers_modules_1_parameters_weight_,
            l_self_modules_head_modules_deconv_layers_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_372 = (
            l_self_modules_head_modules_deconv_layers_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_1_buffers_running_var_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_1_parameters_weight_
        ) = l_self_modules_head_modules_deconv_layers_modules_1_parameters_bias_ = None
        input_374 = torch.nn.functional.relu(input_373, inplace=True)
        input_373 = None
        input_375 = torch.conv_transpose2d(
            input_374,
            l_self_modules_head_modules_deconv_layers_modules_3_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (0, 0),
            1,
            (1, 1),
        )
        input_374 = (
            l_self_modules_head_modules_deconv_layers_modules_3_parameters_weight_
        ) = None
        input_376 = torch.nn.functional.batch_norm(
            input_375,
            l_self_modules_head_modules_deconv_layers_modules_4_buffers_running_mean_,
            l_self_modules_head_modules_deconv_layers_modules_4_buffers_running_var_,
            l_self_modules_head_modules_deconv_layers_modules_4_parameters_weight_,
            l_self_modules_head_modules_deconv_layers_modules_4_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_375 = (
            l_self_modules_head_modules_deconv_layers_modules_4_buffers_running_mean_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_4_buffers_running_var_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_4_parameters_weight_
        ) = l_self_modules_head_modules_deconv_layers_modules_4_parameters_bias_ = None
        input_377 = torch.nn.functional.relu(input_376, inplace=True)
        input_376 = None
        input_378 = torch.conv_transpose2d(
            input_377,
            l_self_modules_head_modules_deconv_layers_modules_6_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (0, 0),
            1,
            (1, 1),
        )
        input_377 = (
            l_self_modules_head_modules_deconv_layers_modules_6_parameters_weight_
        ) = None
        input_379 = torch.nn.functional.batch_norm(
            input_378,
            l_self_modules_head_modules_deconv_layers_modules_7_buffers_running_mean_,
            l_self_modules_head_modules_deconv_layers_modules_7_buffers_running_var_,
            l_self_modules_head_modules_deconv_layers_modules_7_parameters_weight_,
            l_self_modules_head_modules_deconv_layers_modules_7_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_378 = (
            l_self_modules_head_modules_deconv_layers_modules_7_buffers_running_mean_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_7_buffers_running_var_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_7_parameters_weight_
        ) = l_self_modules_head_modules_deconv_layers_modules_7_parameters_bias_ = None
        input_380 = torch.nn.functional.relu(input_379, inplace=True)
        input_379 = None
        x_4 = torch.conv2d(
            input_380,
            l_self_modules_head_modules_final_layer_parameters_weight_,
            l_self_modules_head_modules_final_layer_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_380 = (
            l_self_modules_head_modules_final_layer_parameters_weight_
        ) = l_self_modules_head_modules_final_layer_parameters_bias_ = None
        return (x_4,)
