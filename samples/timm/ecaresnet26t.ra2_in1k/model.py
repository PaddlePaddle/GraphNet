import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_conv1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        s1: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_conv1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_conv1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_conv1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_conv1_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv1_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_conv1_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_conv1_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv1_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_conv1_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_downsample_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_downsample_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_downsample_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_downsample_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_downsample_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_downsample_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_downsample_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_downsample_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_downsample_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_downsample_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_downsample_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_downsample_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_downsample_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_downsample_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_downsample_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_downsample_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_se_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_fc_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_conv1_modules_0_parameters_weight_ = (
            L_self_modules_conv1_modules_0_parameters_weight_
        )
        l_x_ = L_x_
        l_self_modules_conv1_modules_1_buffers_running_mean_ = (
            L_self_modules_conv1_modules_1_buffers_running_mean_
        )
        l_self_modules_conv1_modules_1_buffers_running_var_ = (
            L_self_modules_conv1_modules_1_buffers_running_var_
        )
        l_self_modules_conv1_modules_1_parameters_weight_ = (
            L_self_modules_conv1_modules_1_parameters_weight_
        )
        l_self_modules_conv1_modules_1_parameters_bias_ = (
            L_self_modules_conv1_modules_1_parameters_bias_
        )
        l_self_modules_conv1_modules_3_parameters_weight_ = (
            L_self_modules_conv1_modules_3_parameters_weight_
        )
        l_self_modules_conv1_modules_4_buffers_running_mean_ = (
            L_self_modules_conv1_modules_4_buffers_running_mean_
        )
        l_self_modules_conv1_modules_4_buffers_running_var_ = (
            L_self_modules_conv1_modules_4_buffers_running_var_
        )
        l_self_modules_conv1_modules_4_parameters_weight_ = (
            L_self_modules_conv1_modules_4_parameters_weight_
        )
        l_self_modules_conv1_modules_4_parameters_bias_ = (
            L_self_modules_conv1_modules_4_parameters_bias_
        )
        l_self_modules_conv1_modules_6_parameters_weight_ = (
            L_self_modules_conv1_modules_6_parameters_weight_
        )
        l_self_modules_bn1_buffers_running_mean_ = (
            L_self_modules_bn1_buffers_running_mean_
        )
        l_self_modules_bn1_buffers_running_var_ = (
            L_self_modules_bn1_buffers_running_var_
        )
        l_self_modules_bn1_parameters_weight_ = L_self_modules_bn1_parameters_weight_
        l_self_modules_bn1_parameters_bias_ = L_self_modules_bn1_parameters_bias_
        l_self_modules_layer1_modules_0_modules_conv1_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_conv1_parameters_weight_
        )
        l_self_modules_layer1_modules_0_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer1_modules_0_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer1_modules_0_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer1_modules_0_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer1_modules_0_modules_bn1_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_bn1_parameters_weight_
        )
        l_self_modules_layer1_modules_0_modules_bn1_parameters_bias_ = (
            L_self_modules_layer1_modules_0_modules_bn1_parameters_bias_
        )
        l_self_modules_layer1_modules_0_modules_conv2_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_conv2_parameters_weight_
        )
        l_self_modules_layer1_modules_0_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer1_modules_0_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer1_modules_0_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer1_modules_0_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer1_modules_0_modules_bn2_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_bn2_parameters_weight_
        )
        l_self_modules_layer1_modules_0_modules_bn2_parameters_bias_ = (
            L_self_modules_layer1_modules_0_modules_bn2_parameters_bias_
        )
        l_self_modules_layer1_modules_0_modules_conv3_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_conv3_parameters_weight_
        )
        l_self_modules_layer1_modules_0_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer1_modules_0_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer1_modules_0_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer1_modules_0_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer1_modules_0_modules_bn3_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_bn3_parameters_weight_
        )
        l_self_modules_layer1_modules_0_modules_bn3_parameters_bias_ = (
            L_self_modules_layer1_modules_0_modules_bn3_parameters_bias_
        )
        l_self_modules_layer1_modules_0_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_se_modules_conv_parameters_weight_
        )
        l_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_layer1_modules_0_modules_downsample_modules_2_buffers_running_mean_ = L_self_modules_layer1_modules_0_modules_downsample_modules_2_buffers_running_mean_
        l_self_modules_layer1_modules_0_modules_downsample_modules_2_buffers_running_var_ = L_self_modules_layer1_modules_0_modules_downsample_modules_2_buffers_running_var_
        l_self_modules_layer1_modules_0_modules_downsample_modules_2_parameters_weight_ = L_self_modules_layer1_modules_0_modules_downsample_modules_2_parameters_weight_
        l_self_modules_layer1_modules_0_modules_downsample_modules_2_parameters_bias_ = L_self_modules_layer1_modules_0_modules_downsample_modules_2_parameters_bias_
        l_self_modules_layer1_modules_1_modules_conv1_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_conv1_parameters_weight_
        )
        l_self_modules_layer1_modules_1_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer1_modules_1_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer1_modules_1_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer1_modules_1_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer1_modules_1_modules_bn1_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_bn1_parameters_weight_
        )
        l_self_modules_layer1_modules_1_modules_bn1_parameters_bias_ = (
            L_self_modules_layer1_modules_1_modules_bn1_parameters_bias_
        )
        l_self_modules_layer1_modules_1_modules_conv2_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_conv2_parameters_weight_
        )
        l_self_modules_layer1_modules_1_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer1_modules_1_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer1_modules_1_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer1_modules_1_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer1_modules_1_modules_bn2_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_bn2_parameters_weight_
        )
        l_self_modules_layer1_modules_1_modules_bn2_parameters_bias_ = (
            L_self_modules_layer1_modules_1_modules_bn2_parameters_bias_
        )
        l_self_modules_layer1_modules_1_modules_conv3_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_conv3_parameters_weight_
        )
        l_self_modules_layer1_modules_1_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer1_modules_1_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer1_modules_1_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer1_modules_1_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer1_modules_1_modules_bn3_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_bn3_parameters_weight_
        )
        l_self_modules_layer1_modules_1_modules_bn3_parameters_bias_ = (
            L_self_modules_layer1_modules_1_modules_bn3_parameters_bias_
        )
        l_self_modules_layer1_modules_1_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_se_modules_conv_parameters_weight_
        )
        l_self_modules_layer2_modules_0_modules_conv1_parameters_weight_ = (
            L_self_modules_layer2_modules_0_modules_conv1_parameters_weight_
        )
        l_self_modules_layer2_modules_0_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer2_modules_0_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer2_modules_0_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer2_modules_0_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer2_modules_0_modules_bn1_parameters_weight_ = (
            L_self_modules_layer2_modules_0_modules_bn1_parameters_weight_
        )
        l_self_modules_layer2_modules_0_modules_bn1_parameters_bias_ = (
            L_self_modules_layer2_modules_0_modules_bn1_parameters_bias_
        )
        l_self_modules_layer2_modules_0_modules_conv2_parameters_weight_ = (
            L_self_modules_layer2_modules_0_modules_conv2_parameters_weight_
        )
        l_self_modules_layer2_modules_0_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer2_modules_0_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer2_modules_0_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer2_modules_0_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer2_modules_0_modules_bn2_parameters_weight_ = (
            L_self_modules_layer2_modules_0_modules_bn2_parameters_weight_
        )
        l_self_modules_layer2_modules_0_modules_bn2_parameters_bias_ = (
            L_self_modules_layer2_modules_0_modules_bn2_parameters_bias_
        )
        l_self_modules_layer2_modules_0_modules_conv3_parameters_weight_ = (
            L_self_modules_layer2_modules_0_modules_conv3_parameters_weight_
        )
        l_self_modules_layer2_modules_0_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer2_modules_0_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer2_modules_0_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer2_modules_0_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer2_modules_0_modules_bn3_parameters_weight_ = (
            L_self_modules_layer2_modules_0_modules_bn3_parameters_weight_
        )
        l_self_modules_layer2_modules_0_modules_bn3_parameters_bias_ = (
            L_self_modules_layer2_modules_0_modules_bn3_parameters_bias_
        )
        l_self_modules_layer2_modules_0_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer2_modules_0_modules_se_modules_conv_parameters_weight_
        )
        l_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_layer2_modules_0_modules_downsample_modules_2_buffers_running_mean_ = L_self_modules_layer2_modules_0_modules_downsample_modules_2_buffers_running_mean_
        l_self_modules_layer2_modules_0_modules_downsample_modules_2_buffers_running_var_ = L_self_modules_layer2_modules_0_modules_downsample_modules_2_buffers_running_var_
        l_self_modules_layer2_modules_0_modules_downsample_modules_2_parameters_weight_ = L_self_modules_layer2_modules_0_modules_downsample_modules_2_parameters_weight_
        l_self_modules_layer2_modules_0_modules_downsample_modules_2_parameters_bias_ = L_self_modules_layer2_modules_0_modules_downsample_modules_2_parameters_bias_
        l_self_modules_layer2_modules_1_modules_conv1_parameters_weight_ = (
            L_self_modules_layer2_modules_1_modules_conv1_parameters_weight_
        )
        l_self_modules_layer2_modules_1_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer2_modules_1_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer2_modules_1_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer2_modules_1_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer2_modules_1_modules_bn1_parameters_weight_ = (
            L_self_modules_layer2_modules_1_modules_bn1_parameters_weight_
        )
        l_self_modules_layer2_modules_1_modules_bn1_parameters_bias_ = (
            L_self_modules_layer2_modules_1_modules_bn1_parameters_bias_
        )
        l_self_modules_layer2_modules_1_modules_conv2_parameters_weight_ = (
            L_self_modules_layer2_modules_1_modules_conv2_parameters_weight_
        )
        l_self_modules_layer2_modules_1_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer2_modules_1_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer2_modules_1_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer2_modules_1_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer2_modules_1_modules_bn2_parameters_weight_ = (
            L_self_modules_layer2_modules_1_modules_bn2_parameters_weight_
        )
        l_self_modules_layer2_modules_1_modules_bn2_parameters_bias_ = (
            L_self_modules_layer2_modules_1_modules_bn2_parameters_bias_
        )
        l_self_modules_layer2_modules_1_modules_conv3_parameters_weight_ = (
            L_self_modules_layer2_modules_1_modules_conv3_parameters_weight_
        )
        l_self_modules_layer2_modules_1_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer2_modules_1_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer2_modules_1_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer2_modules_1_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer2_modules_1_modules_bn3_parameters_weight_ = (
            L_self_modules_layer2_modules_1_modules_bn3_parameters_weight_
        )
        l_self_modules_layer2_modules_1_modules_bn3_parameters_bias_ = (
            L_self_modules_layer2_modules_1_modules_bn3_parameters_bias_
        )
        l_self_modules_layer2_modules_1_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer2_modules_1_modules_se_modules_conv_parameters_weight_
        )
        l_self_modules_layer3_modules_0_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_0_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_0_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_0_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_0_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_0_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_0_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_0_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_0_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_0_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_0_modules_conv2_parameters_weight_ = (
            L_self_modules_layer3_modules_0_modules_conv2_parameters_weight_
        )
        l_self_modules_layer3_modules_0_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_0_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_0_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer3_modules_0_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer3_modules_0_modules_bn2_parameters_weight_ = (
            L_self_modules_layer3_modules_0_modules_bn2_parameters_weight_
        )
        l_self_modules_layer3_modules_0_modules_bn2_parameters_bias_ = (
            L_self_modules_layer3_modules_0_modules_bn2_parameters_bias_
        )
        l_self_modules_layer3_modules_0_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_0_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_0_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_0_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_0_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_0_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_0_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_0_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_0_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_0_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_0_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer3_modules_0_modules_se_modules_conv_parameters_weight_
        )
        l_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_layer3_modules_0_modules_downsample_modules_2_buffers_running_mean_ = L_self_modules_layer3_modules_0_modules_downsample_modules_2_buffers_running_mean_
        l_self_modules_layer3_modules_0_modules_downsample_modules_2_buffers_running_var_ = L_self_modules_layer3_modules_0_modules_downsample_modules_2_buffers_running_var_
        l_self_modules_layer3_modules_0_modules_downsample_modules_2_parameters_weight_ = L_self_modules_layer3_modules_0_modules_downsample_modules_2_parameters_weight_
        l_self_modules_layer3_modules_0_modules_downsample_modules_2_parameters_bias_ = L_self_modules_layer3_modules_0_modules_downsample_modules_2_parameters_bias_
        l_self_modules_layer3_modules_1_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_1_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_1_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_1_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_1_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_1_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_1_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_1_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_1_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_1_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_1_modules_conv2_parameters_weight_ = (
            L_self_modules_layer3_modules_1_modules_conv2_parameters_weight_
        )
        l_self_modules_layer3_modules_1_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_1_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_1_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer3_modules_1_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer3_modules_1_modules_bn2_parameters_weight_ = (
            L_self_modules_layer3_modules_1_modules_bn2_parameters_weight_
        )
        l_self_modules_layer3_modules_1_modules_bn2_parameters_bias_ = (
            L_self_modules_layer3_modules_1_modules_bn2_parameters_bias_
        )
        l_self_modules_layer3_modules_1_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_1_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_1_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_1_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_1_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_1_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_1_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_1_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_1_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_1_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_1_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer3_modules_1_modules_se_modules_conv_parameters_weight_
        )
        l_self_modules_layer4_modules_0_modules_conv1_parameters_weight_ = (
            L_self_modules_layer4_modules_0_modules_conv1_parameters_weight_
        )
        l_self_modules_layer4_modules_0_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer4_modules_0_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer4_modules_0_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer4_modules_0_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer4_modules_0_modules_bn1_parameters_weight_ = (
            L_self_modules_layer4_modules_0_modules_bn1_parameters_weight_
        )
        l_self_modules_layer4_modules_0_modules_bn1_parameters_bias_ = (
            L_self_modules_layer4_modules_0_modules_bn1_parameters_bias_
        )
        l_self_modules_layer4_modules_0_modules_conv2_parameters_weight_ = (
            L_self_modules_layer4_modules_0_modules_conv2_parameters_weight_
        )
        l_self_modules_layer4_modules_0_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer4_modules_0_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer4_modules_0_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer4_modules_0_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer4_modules_0_modules_bn2_parameters_weight_ = (
            L_self_modules_layer4_modules_0_modules_bn2_parameters_weight_
        )
        l_self_modules_layer4_modules_0_modules_bn2_parameters_bias_ = (
            L_self_modules_layer4_modules_0_modules_bn2_parameters_bias_
        )
        l_self_modules_layer4_modules_0_modules_conv3_parameters_weight_ = (
            L_self_modules_layer4_modules_0_modules_conv3_parameters_weight_
        )
        l_self_modules_layer4_modules_0_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer4_modules_0_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer4_modules_0_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer4_modules_0_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer4_modules_0_modules_bn3_parameters_weight_ = (
            L_self_modules_layer4_modules_0_modules_bn3_parameters_weight_
        )
        l_self_modules_layer4_modules_0_modules_bn3_parameters_bias_ = (
            L_self_modules_layer4_modules_0_modules_bn3_parameters_bias_
        )
        l_self_modules_layer4_modules_0_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer4_modules_0_modules_se_modules_conv_parameters_weight_
        )
        l_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_layer4_modules_0_modules_downsample_modules_2_buffers_running_mean_ = L_self_modules_layer4_modules_0_modules_downsample_modules_2_buffers_running_mean_
        l_self_modules_layer4_modules_0_modules_downsample_modules_2_buffers_running_var_ = L_self_modules_layer4_modules_0_modules_downsample_modules_2_buffers_running_var_
        l_self_modules_layer4_modules_0_modules_downsample_modules_2_parameters_weight_ = L_self_modules_layer4_modules_0_modules_downsample_modules_2_parameters_weight_
        l_self_modules_layer4_modules_0_modules_downsample_modules_2_parameters_bias_ = L_self_modules_layer4_modules_0_modules_downsample_modules_2_parameters_bias_
        l_self_modules_layer4_modules_1_modules_conv1_parameters_weight_ = (
            L_self_modules_layer4_modules_1_modules_conv1_parameters_weight_
        )
        l_self_modules_layer4_modules_1_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer4_modules_1_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer4_modules_1_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer4_modules_1_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer4_modules_1_modules_bn1_parameters_weight_ = (
            L_self_modules_layer4_modules_1_modules_bn1_parameters_weight_
        )
        l_self_modules_layer4_modules_1_modules_bn1_parameters_bias_ = (
            L_self_modules_layer4_modules_1_modules_bn1_parameters_bias_
        )
        l_self_modules_layer4_modules_1_modules_conv2_parameters_weight_ = (
            L_self_modules_layer4_modules_1_modules_conv2_parameters_weight_
        )
        l_self_modules_layer4_modules_1_modules_bn2_buffers_running_mean_ = (
            L_self_modules_layer4_modules_1_modules_bn2_buffers_running_mean_
        )
        l_self_modules_layer4_modules_1_modules_bn2_buffers_running_var_ = (
            L_self_modules_layer4_modules_1_modules_bn2_buffers_running_var_
        )
        l_self_modules_layer4_modules_1_modules_bn2_parameters_weight_ = (
            L_self_modules_layer4_modules_1_modules_bn2_parameters_weight_
        )
        l_self_modules_layer4_modules_1_modules_bn2_parameters_bias_ = (
            L_self_modules_layer4_modules_1_modules_bn2_parameters_bias_
        )
        l_self_modules_layer4_modules_1_modules_conv3_parameters_weight_ = (
            L_self_modules_layer4_modules_1_modules_conv3_parameters_weight_
        )
        l_self_modules_layer4_modules_1_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer4_modules_1_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer4_modules_1_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer4_modules_1_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer4_modules_1_modules_bn3_parameters_weight_ = (
            L_self_modules_layer4_modules_1_modules_bn3_parameters_weight_
        )
        l_self_modules_layer4_modules_1_modules_bn3_parameters_bias_ = (
            L_self_modules_layer4_modules_1_modules_bn3_parameters_bias_
        )
        l_self_modules_layer4_modules_1_modules_se_modules_conv_parameters_weight_ = (
            L_self_modules_layer4_modules_1_modules_se_modules_conv_parameters_weight_
        )
        l_self_modules_fc_parameters_weight_ = L_self_modules_fc_parameters_weight_
        l_self_modules_fc_parameters_bias_ = L_self_modules_fc_parameters_bias_
        input_1 = torch.conv2d(
            l_x_,
            l_self_modules_conv1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_x_ = l_self_modules_conv1_modules_0_parameters_weight_ = None
        input_2 = torch.nn.functional.batch_norm(
            input_1,
            l_self_modules_conv1_modules_1_buffers_running_mean_,
            l_self_modules_conv1_modules_1_buffers_running_var_,
            l_self_modules_conv1_modules_1_parameters_weight_,
            l_self_modules_conv1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_1 = (
            l_self_modules_conv1_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_conv1_modules_1_buffers_running_var_
        ) = (
            l_self_modules_conv1_modules_1_parameters_weight_
        ) = l_self_modules_conv1_modules_1_parameters_bias_ = None
        input_3 = torch.nn.functional.relu(input_2, inplace=True)
        input_2 = None
        input_4 = torch.conv2d(
            input_3,
            l_self_modules_conv1_modules_3_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_3 = l_self_modules_conv1_modules_3_parameters_weight_ = None
        input_5 = torch.nn.functional.batch_norm(
            input_4,
            l_self_modules_conv1_modules_4_buffers_running_mean_,
            l_self_modules_conv1_modules_4_buffers_running_var_,
            l_self_modules_conv1_modules_4_parameters_weight_,
            l_self_modules_conv1_modules_4_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_4 = (
            l_self_modules_conv1_modules_4_buffers_running_mean_
        ) = (
            l_self_modules_conv1_modules_4_buffers_running_var_
        ) = (
            l_self_modules_conv1_modules_4_parameters_weight_
        ) = l_self_modules_conv1_modules_4_parameters_bias_ = None
        input_6 = torch.nn.functional.relu(input_5, inplace=True)
        input_5 = None
        input_7 = torch.conv2d(
            input_6,
            l_self_modules_conv1_modules_6_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_6 = l_self_modules_conv1_modules_6_parameters_weight_ = None
        x = torch.nn.functional.batch_norm(
            input_7,
            l_self_modules_bn1_buffers_running_mean_,
            l_self_modules_bn1_buffers_running_var_,
            l_self_modules_bn1_parameters_weight_,
            l_self_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_7 = (
            l_self_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_bn1_parameters_weight_
        ) = l_self_modules_bn1_parameters_bias_ = None
        x_1 = torch.nn.functional.relu(x, inplace=True)
        x = None
        x_2 = torch.nn.functional.max_pool2d(
            x_1, 3, 2, 1, 1, ceil_mode=False, return_indices=False
        )
        x_1 = None
        x_3 = torch.conv2d(
            x_2,
            l_self_modules_layer1_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer1_modules_0_modules_conv1_parameters_weight_ = None
        x_4 = torch.nn.functional.batch_norm(
            x_3,
            l_self_modules_layer1_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_layer1_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_layer1_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_layer1_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_3 = (
            l_self_modules_layer1_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_0_modules_bn1_parameters_weight_
        ) = l_self_modules_layer1_modules_0_modules_bn1_parameters_bias_ = None
        x_5 = torch.nn.functional.relu(x_4, inplace=True)
        x_4 = None
        x_6 = torch.conv2d(
            x_5,
            l_self_modules_layer1_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_5 = l_self_modules_layer1_modules_0_modules_conv2_parameters_weight_ = None
        x_7 = torch.nn.functional.batch_norm(
            x_6,
            l_self_modules_layer1_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_layer1_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_layer1_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_layer1_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_6 = (
            l_self_modules_layer1_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_0_modules_bn2_parameters_weight_
        ) = l_self_modules_layer1_modules_0_modules_bn2_parameters_bias_ = None
        x_8 = torch.nn.functional.relu(x_7, inplace=True)
        x_7 = None
        x_9 = torch.conv2d(
            x_8,
            l_self_modules_layer1_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_8 = l_self_modules_layer1_modules_0_modules_conv3_parameters_weight_ = None
        x_10 = torch.nn.functional.batch_norm(
            x_9,
            l_self_modules_layer1_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_layer1_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_layer1_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_layer1_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_9 = (
            l_self_modules_layer1_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_0_modules_bn3_parameters_weight_
        ) = l_self_modules_layer1_modules_0_modules_bn3_parameters_bias_ = None
        mean = x_10.mean((2, 3))
        sym_sum = torch.sym_sum([-1, s1])
        s1 = None
        floordiv = sym_sum // 4
        sym_sum_1 = torch.sym_sum([1, floordiv])
        floordiv = sym_sum_1 = None
        y = mean.view(1, 1, -1)
        mean = None
        y_1 = torch.conv1d(
            y,
            l_self_modules_layer1_modules_0_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y = (
            l_self_modules_layer1_modules_0_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid = y_1.sigmoid()
        y_1 = None
        y_2 = sigmoid.view(1, -1, 1, 1)
        sigmoid = None
        expand_as = y_2.expand_as(x_10)
        y_2 = None
        x_11 = x_10 * expand_as
        x_10 = expand_as = None
        input_8 = torch.conv2d(
            x_2,
            l_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_2 = l_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_ = (None)
        input_9 = torch.nn.functional.batch_norm(
            input_8,
            l_self_modules_layer1_modules_0_modules_downsample_modules_2_buffers_running_mean_,
            l_self_modules_layer1_modules_0_modules_downsample_modules_2_buffers_running_var_,
            l_self_modules_layer1_modules_0_modules_downsample_modules_2_parameters_weight_,
            l_self_modules_layer1_modules_0_modules_downsample_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_8 = l_self_modules_layer1_modules_0_modules_downsample_modules_2_buffers_running_mean_ = l_self_modules_layer1_modules_0_modules_downsample_modules_2_buffers_running_var_ = l_self_modules_layer1_modules_0_modules_downsample_modules_2_parameters_weight_ = l_self_modules_layer1_modules_0_modules_downsample_modules_2_parameters_bias_ = (None)
        x_11 += input_9
        x_12 = x_11
        x_11 = input_9 = None
        x_13 = torch.nn.functional.relu(x_12, inplace=True)
        x_12 = None
        x_14 = torch.conv2d(
            x_13,
            l_self_modules_layer1_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer1_modules_1_modules_conv1_parameters_weight_ = None
        x_15 = torch.nn.functional.batch_norm(
            x_14,
            l_self_modules_layer1_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_layer1_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_layer1_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_layer1_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_14 = (
            l_self_modules_layer1_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_1_modules_bn1_parameters_weight_
        ) = l_self_modules_layer1_modules_1_modules_bn1_parameters_bias_ = None
        x_16 = torch.nn.functional.relu(x_15, inplace=True)
        x_15 = None
        x_17 = torch.conv2d(
            x_16,
            l_self_modules_layer1_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_16 = l_self_modules_layer1_modules_1_modules_conv2_parameters_weight_ = None
        x_18 = torch.nn.functional.batch_norm(
            x_17,
            l_self_modules_layer1_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_layer1_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_layer1_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_layer1_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_17 = (
            l_self_modules_layer1_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_1_modules_bn2_parameters_weight_
        ) = l_self_modules_layer1_modules_1_modules_bn2_parameters_bias_ = None
        x_19 = torch.nn.functional.relu(x_18, inplace=True)
        x_18 = None
        x_20 = torch.conv2d(
            x_19,
            l_self_modules_layer1_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_19 = l_self_modules_layer1_modules_1_modules_conv3_parameters_weight_ = None
        x_21 = torch.nn.functional.batch_norm(
            x_20,
            l_self_modules_layer1_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_layer1_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_layer1_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_layer1_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_20 = (
            l_self_modules_layer1_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_1_modules_bn3_parameters_weight_
        ) = l_self_modules_layer1_modules_1_modules_bn3_parameters_bias_ = None
        mean_1 = x_21.mean((2, 3))
        y_3 = mean_1.view(1, 1, -1)
        mean_1 = None
        y_4 = torch.conv1d(
            y_3,
            l_self_modules_layer1_modules_1_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_3 = (
            l_self_modules_layer1_modules_1_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_1 = y_4.sigmoid()
        y_4 = None
        y_5 = sigmoid_1.view(1, -1, 1, 1)
        sigmoid_1 = None
        expand_as_1 = y_5.expand_as(x_21)
        y_5 = None
        x_22 = x_21 * expand_as_1
        x_21 = expand_as_1 = None
        x_22 += x_13
        x_23 = x_22
        x_22 = x_13 = None
        x_24 = torch.nn.functional.relu(x_23, inplace=True)
        x_23 = None
        x_25 = torch.conv2d(
            x_24,
            l_self_modules_layer2_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_0_modules_conv1_parameters_weight_ = None
        x_26 = torch.nn.functional.batch_norm(
            x_25,
            l_self_modules_layer2_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_25 = (
            l_self_modules_layer2_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_0_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_0_modules_bn1_parameters_bias_ = None
        x_27 = torch.nn.functional.relu(x_26, inplace=True)
        x_26 = None
        x_28 = torch.conv2d(
            x_27,
            l_self_modules_layer2_modules_0_modules_conv2_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_27 = l_self_modules_layer2_modules_0_modules_conv2_parameters_weight_ = None
        x_29 = torch.nn.functional.batch_norm(
            x_28,
            l_self_modules_layer2_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_28 = (
            l_self_modules_layer2_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_0_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_0_modules_bn2_parameters_bias_ = None
        x_30 = torch.nn.functional.relu(x_29, inplace=True)
        x_29 = None
        x_31 = torch.conv2d(
            x_30,
            l_self_modules_layer2_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_30 = l_self_modules_layer2_modules_0_modules_conv3_parameters_weight_ = None
        x_32 = torch.nn.functional.batch_norm(
            x_31,
            l_self_modules_layer2_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_31 = (
            l_self_modules_layer2_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_0_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_0_modules_bn3_parameters_bias_ = None
        mean_2 = x_32.mean((2, 3))
        floordiv_1 = sym_sum // 8
        sym_sum_2 = torch.sym_sum([1, floordiv_1])
        floordiv_1 = sym_sum_2 = None
        y_6 = mean_2.view(1, 1, -1)
        mean_2 = None
        y_7 = torch.conv1d(
            y_6,
            l_self_modules_layer2_modules_0_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_6 = (
            l_self_modules_layer2_modules_0_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_2 = y_7.sigmoid()
        y_7 = None
        y_8 = sigmoid_2.view(1, -1, 1, 1)
        sigmoid_2 = None
        expand_as_2 = y_8.expand_as(x_32)
        y_8 = None
        x_33 = x_32 * expand_as_2
        x_32 = expand_as_2 = None
        input_10 = torch._C._nn.avg_pool2d(x_24, 2, 2, 0, True, False, None)
        x_24 = None
        input_11 = torch.conv2d(
            input_10,
            l_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_10 = l_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_ = (None)
        input_12 = torch.nn.functional.batch_norm(
            input_11,
            l_self_modules_layer2_modules_0_modules_downsample_modules_2_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_downsample_modules_2_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_downsample_modules_2_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_downsample_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_11 = l_self_modules_layer2_modules_0_modules_downsample_modules_2_buffers_running_mean_ = l_self_modules_layer2_modules_0_modules_downsample_modules_2_buffers_running_var_ = l_self_modules_layer2_modules_0_modules_downsample_modules_2_parameters_weight_ = l_self_modules_layer2_modules_0_modules_downsample_modules_2_parameters_bias_ = (None)
        x_33 += input_12
        x_34 = x_33
        x_33 = input_12 = None
        x_35 = torch.nn.functional.relu(x_34, inplace=True)
        x_34 = None
        x_36 = torch.conv2d(
            x_35,
            l_self_modules_layer2_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_1_modules_conv1_parameters_weight_ = None
        x_37 = torch.nn.functional.batch_norm(
            x_36,
            l_self_modules_layer2_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_36 = (
            l_self_modules_layer2_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_1_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_1_modules_bn1_parameters_bias_ = None
        x_38 = torch.nn.functional.relu(x_37, inplace=True)
        x_37 = None
        x_39 = torch.conv2d(
            x_38,
            l_self_modules_layer2_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_38 = l_self_modules_layer2_modules_1_modules_conv2_parameters_weight_ = None
        x_40 = torch.nn.functional.batch_norm(
            x_39,
            l_self_modules_layer2_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_layer2_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_layer2_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_39 = (
            l_self_modules_layer2_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_1_modules_bn2_parameters_weight_
        ) = l_self_modules_layer2_modules_1_modules_bn2_parameters_bias_ = None
        x_41 = torch.nn.functional.relu(x_40, inplace=True)
        x_40 = None
        x_42 = torch.conv2d(
            x_41,
            l_self_modules_layer2_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_41 = l_self_modules_layer2_modules_1_modules_conv3_parameters_weight_ = None
        x_43 = torch.nn.functional.batch_norm(
            x_42,
            l_self_modules_layer2_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_42 = (
            l_self_modules_layer2_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_1_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_1_modules_bn3_parameters_bias_ = None
        mean_3 = x_43.mean((2, 3))
        y_9 = mean_3.view(1, 1, -1)
        mean_3 = None
        y_10 = torch.conv1d(
            y_9,
            l_self_modules_layer2_modules_1_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_9 = (
            l_self_modules_layer2_modules_1_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_3 = y_10.sigmoid()
        y_10 = None
        y_11 = sigmoid_3.view(1, -1, 1, 1)
        sigmoid_3 = None
        expand_as_3 = y_11.expand_as(x_43)
        y_11 = None
        x_44 = x_43 * expand_as_3
        x_43 = expand_as_3 = None
        x_44 += x_35
        x_45 = x_44
        x_44 = x_35 = None
        x_46 = torch.nn.functional.relu(x_45, inplace=True)
        x_45 = None
        x_47 = torch.conv2d(
            x_46,
            l_self_modules_layer3_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_0_modules_conv1_parameters_weight_ = None
        x_48 = torch.nn.functional.batch_norm(
            x_47,
            l_self_modules_layer3_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_47 = (
            l_self_modules_layer3_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_0_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_0_modules_bn1_parameters_bias_ = None
        x_49 = torch.nn.functional.relu(x_48, inplace=True)
        x_48 = None
        x_50 = torch.conv2d(
            x_49,
            l_self_modules_layer3_modules_0_modules_conv2_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_49 = l_self_modules_layer3_modules_0_modules_conv2_parameters_weight_ = None
        x_51 = torch.nn.functional.batch_norm(
            x_50,
            l_self_modules_layer3_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_50 = (
            l_self_modules_layer3_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_0_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_0_modules_bn2_parameters_bias_ = None
        x_52 = torch.nn.functional.relu(x_51, inplace=True)
        x_51 = None
        x_53 = torch.conv2d(
            x_52,
            l_self_modules_layer3_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_52 = l_self_modules_layer3_modules_0_modules_conv3_parameters_weight_ = None
        x_54 = torch.nn.functional.batch_norm(
            x_53,
            l_self_modules_layer3_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_53 = (
            l_self_modules_layer3_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_0_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_0_modules_bn3_parameters_bias_ = None
        mean_4 = x_54.mean((2, 3))
        floordiv_2 = sym_sum // 16
        sym_sum_3 = torch.sym_sum([1, floordiv_2])
        floordiv_2 = sym_sum_3 = None
        y_12 = mean_4.view(1, 1, -1)
        mean_4 = None
        y_13 = torch.conv1d(
            y_12,
            l_self_modules_layer3_modules_0_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_12 = (
            l_self_modules_layer3_modules_0_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_4 = y_13.sigmoid()
        y_13 = None
        y_14 = sigmoid_4.view(1, -1, 1, 1)
        sigmoid_4 = None
        expand_as_4 = y_14.expand_as(x_54)
        y_14 = None
        x_55 = x_54 * expand_as_4
        x_54 = expand_as_4 = None
        input_13 = torch._C._nn.avg_pool2d(x_46, 2, 2, 0, True, False, None)
        x_46 = None
        input_14 = torch.conv2d(
            input_13,
            l_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_13 = l_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_ = (None)
        input_15 = torch.nn.functional.batch_norm(
            input_14,
            l_self_modules_layer3_modules_0_modules_downsample_modules_2_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_downsample_modules_2_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_downsample_modules_2_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_downsample_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_14 = l_self_modules_layer3_modules_0_modules_downsample_modules_2_buffers_running_mean_ = l_self_modules_layer3_modules_0_modules_downsample_modules_2_buffers_running_var_ = l_self_modules_layer3_modules_0_modules_downsample_modules_2_parameters_weight_ = l_self_modules_layer3_modules_0_modules_downsample_modules_2_parameters_bias_ = (None)
        x_55 += input_15
        x_56 = x_55
        x_55 = input_15 = None
        x_57 = torch.nn.functional.relu(x_56, inplace=True)
        x_56 = None
        x_58 = torch.conv2d(
            x_57,
            l_self_modules_layer3_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_1_modules_conv1_parameters_weight_ = None
        x_59 = torch.nn.functional.batch_norm(
            x_58,
            l_self_modules_layer3_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_58 = (
            l_self_modules_layer3_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_1_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_1_modules_bn1_parameters_bias_ = None
        x_60 = torch.nn.functional.relu(x_59, inplace=True)
        x_59 = None
        x_61 = torch.conv2d(
            x_60,
            l_self_modules_layer3_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_60 = l_self_modules_layer3_modules_1_modules_conv2_parameters_weight_ = None
        x_62 = torch.nn.functional.batch_norm(
            x_61,
            l_self_modules_layer3_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_layer3_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_layer3_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_61 = (
            l_self_modules_layer3_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_1_modules_bn2_parameters_weight_
        ) = l_self_modules_layer3_modules_1_modules_bn2_parameters_bias_ = None
        x_63 = torch.nn.functional.relu(x_62, inplace=True)
        x_62 = None
        x_64 = torch.conv2d(
            x_63,
            l_self_modules_layer3_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_63 = l_self_modules_layer3_modules_1_modules_conv3_parameters_weight_ = None
        x_65 = torch.nn.functional.batch_norm(
            x_64,
            l_self_modules_layer3_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_64 = (
            l_self_modules_layer3_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_1_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_1_modules_bn3_parameters_bias_ = None
        mean_5 = x_65.mean((2, 3))
        y_15 = mean_5.view(1, 1, -1)
        mean_5 = None
        y_16 = torch.conv1d(
            y_15,
            l_self_modules_layer3_modules_1_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (2,),
            (1,),
            1,
        )
        y_15 = (
            l_self_modules_layer3_modules_1_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_5 = y_16.sigmoid()
        y_16 = None
        y_17 = sigmoid_5.view(1, -1, 1, 1)
        sigmoid_5 = None
        expand_as_5 = y_17.expand_as(x_65)
        y_17 = None
        x_66 = x_65 * expand_as_5
        x_65 = expand_as_5 = None
        x_66 += x_57
        x_67 = x_66
        x_66 = x_57 = None
        x_68 = torch.nn.functional.relu(x_67, inplace=True)
        x_67 = None
        x_69 = torch.conv2d(
            x_68,
            l_self_modules_layer4_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer4_modules_0_modules_conv1_parameters_weight_ = None
        x_70 = torch.nn.functional.batch_norm(
            x_69,
            l_self_modules_layer4_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_69 = (
            l_self_modules_layer4_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_0_modules_bn1_parameters_weight_
        ) = l_self_modules_layer4_modules_0_modules_bn1_parameters_bias_ = None
        x_71 = torch.nn.functional.relu(x_70, inplace=True)
        x_70 = None
        x_72 = torch.conv2d(
            x_71,
            l_self_modules_layer4_modules_0_modules_conv2_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_71 = l_self_modules_layer4_modules_0_modules_conv2_parameters_weight_ = None
        x_73 = torch.nn.functional.batch_norm(
            x_72,
            l_self_modules_layer4_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_72 = (
            l_self_modules_layer4_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_0_modules_bn2_parameters_weight_
        ) = l_self_modules_layer4_modules_0_modules_bn2_parameters_bias_ = None
        x_74 = torch.nn.functional.relu(x_73, inplace=True)
        x_73 = None
        x_75 = torch.conv2d(
            x_74,
            l_self_modules_layer4_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_74 = l_self_modules_layer4_modules_0_modules_conv3_parameters_weight_ = None
        x_76 = torch.nn.functional.batch_norm(
            x_75,
            l_self_modules_layer4_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_75 = (
            l_self_modules_layer4_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_0_modules_bn3_parameters_weight_
        ) = l_self_modules_layer4_modules_0_modules_bn3_parameters_bias_ = None
        mean_6 = x_76.mean((2, 3))
        floordiv_3 = sym_sum // 32
        sym_sum = None
        sym_sum_4 = torch.sym_sum([1, floordiv_3])
        floordiv_3 = sym_sum_4 = None
        y_18 = mean_6.view(1, 1, -1)
        mean_6 = None
        y_19 = torch.conv1d(
            y_18,
            l_self_modules_layer4_modules_0_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (3,),
            (1,),
            1,
        )
        y_18 = (
            l_self_modules_layer4_modules_0_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_6 = y_19.sigmoid()
        y_19 = None
        y_20 = sigmoid_6.view(1, -1, 1, 1)
        sigmoid_6 = None
        expand_as_6 = y_20.expand_as(x_76)
        y_20 = None
        x_77 = x_76 * expand_as_6
        x_76 = expand_as_6 = None
        input_16 = torch._C._nn.avg_pool2d(x_68, 2, 2, 0, True, False, None)
        x_68 = None
        input_17 = torch.conv2d(
            input_16,
            l_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_16 = l_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_ = (None)
        input_18 = torch.nn.functional.batch_norm(
            input_17,
            l_self_modules_layer4_modules_0_modules_downsample_modules_2_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_downsample_modules_2_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_downsample_modules_2_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_downsample_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_17 = l_self_modules_layer4_modules_0_modules_downsample_modules_2_buffers_running_mean_ = l_self_modules_layer4_modules_0_modules_downsample_modules_2_buffers_running_var_ = l_self_modules_layer4_modules_0_modules_downsample_modules_2_parameters_weight_ = l_self_modules_layer4_modules_0_modules_downsample_modules_2_parameters_bias_ = (None)
        x_77 += input_18
        x_78 = x_77
        x_77 = input_18 = None
        x_79 = torch.nn.functional.relu(x_78, inplace=True)
        x_78 = None
        x_80 = torch.conv2d(
            x_79,
            l_self_modules_layer4_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer4_modules_1_modules_conv1_parameters_weight_ = None
        x_81 = torch.nn.functional.batch_norm(
            x_80,
            l_self_modules_layer4_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_80 = (
            l_self_modules_layer4_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_1_modules_bn1_parameters_weight_
        ) = l_self_modules_layer4_modules_1_modules_bn1_parameters_bias_ = None
        x_82 = torch.nn.functional.relu(x_81, inplace=True)
        x_81 = None
        x_83 = torch.conv2d(
            x_82,
            l_self_modules_layer4_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_82 = l_self_modules_layer4_modules_1_modules_conv2_parameters_weight_ = None
        x_84 = torch.nn.functional.batch_norm(
            x_83,
            l_self_modules_layer4_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_83 = (
            l_self_modules_layer4_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_1_modules_bn2_parameters_weight_
        ) = l_self_modules_layer4_modules_1_modules_bn2_parameters_bias_ = None
        x_85 = torch.nn.functional.relu(x_84, inplace=True)
        x_84 = None
        x_86 = torch.conv2d(
            x_85,
            l_self_modules_layer4_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_85 = l_self_modules_layer4_modules_1_modules_conv3_parameters_weight_ = None
        x_87 = torch.nn.functional.batch_norm(
            x_86,
            l_self_modules_layer4_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_86 = (
            l_self_modules_layer4_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_1_modules_bn3_parameters_weight_
        ) = l_self_modules_layer4_modules_1_modules_bn3_parameters_bias_ = None
        mean_7 = x_87.mean((2, 3))
        y_21 = mean_7.view(1, 1, -1)
        mean_7 = None
        y_22 = torch.conv1d(
            y_21,
            l_self_modules_layer4_modules_1_modules_se_modules_conv_parameters_weight_,
            None,
            (1,),
            (3,),
            (1,),
            1,
        )
        y_21 = (
            l_self_modules_layer4_modules_1_modules_se_modules_conv_parameters_weight_
        ) = None
        sigmoid_7 = y_22.sigmoid()
        y_22 = None
        y_23 = sigmoid_7.view(1, -1, 1, 1)
        sigmoid_7 = None
        expand_as_7 = y_23.expand_as(x_87)
        y_23 = None
        x_88 = x_87 * expand_as_7
        x_87 = expand_as_7 = None
        x_88 += x_79
        x_89 = x_88
        x_88 = x_79 = None
        x_90 = torch.nn.functional.relu(x_89, inplace=True)
        x_89 = None
        x_91 = torch.nn.functional.adaptive_avg_pool2d(x_90, 1)
        x_90 = None
        x_92 = x_91.flatten(1, -1)
        x_91 = None
        x_93 = torch._C._nn.linear(
            x_92,
            l_self_modules_fc_parameters_weight_,
            l_self_modules_fc_parameters_bias_,
        )
        x_92 = (
            l_self_modules_fc_parameters_weight_
        ) = l_self_modules_fc_parameters_bias_ = None
        return (x_93,)
