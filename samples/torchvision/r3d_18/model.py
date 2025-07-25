import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_stem_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        s1: torch.SymInt,
        s2: torch.SymInt,
        s3: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_stem_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stem_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stem_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_conv1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_conv1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_conv2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_conv2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_conv1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_conv1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_conv2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_conv2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_conv1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_conv1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_conv2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_conv2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_conv1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_conv1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_conv2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_conv2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_conv1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_conv1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_conv2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_conv2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_conv1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_conv1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_conv2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_conv2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_conv1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_conv1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_conv2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_conv2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_conv1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_conv1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_conv2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_conv2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_fc_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_stem_modules_0_parameters_weight_ = (
            L_self_modules_stem_modules_0_parameters_weight_
        )
        l_x_ = L_x_
        l_self_modules_stem_modules_1_buffers_running_mean_ = (
            L_self_modules_stem_modules_1_buffers_running_mean_
        )
        l_self_modules_stem_modules_1_buffers_running_var_ = (
            L_self_modules_stem_modules_1_buffers_running_var_
        )
        l_self_modules_stem_modules_1_parameters_weight_ = (
            L_self_modules_stem_modules_1_parameters_weight_
        )
        l_self_modules_stem_modules_1_parameters_bias_ = (
            L_self_modules_stem_modules_1_parameters_bias_
        )
        l_self_modules_layer1_modules_0_modules_conv1_modules_0_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_conv1_modules_0_parameters_weight_
        )
        l_self_modules_layer1_modules_0_modules_conv1_modules_1_buffers_running_mean_ = L_self_modules_layer1_modules_0_modules_conv1_modules_1_buffers_running_mean_
        l_self_modules_layer1_modules_0_modules_conv1_modules_1_buffers_running_var_ = (
            L_self_modules_layer1_modules_0_modules_conv1_modules_1_buffers_running_var_
        )
        l_self_modules_layer1_modules_0_modules_conv1_modules_1_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_conv1_modules_1_parameters_weight_
        )
        l_self_modules_layer1_modules_0_modules_conv1_modules_1_parameters_bias_ = (
            L_self_modules_layer1_modules_0_modules_conv1_modules_1_parameters_bias_
        )
        l_self_modules_layer1_modules_0_modules_conv2_modules_0_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_conv2_modules_0_parameters_weight_
        )
        l_self_modules_layer1_modules_0_modules_conv2_modules_1_buffers_running_mean_ = L_self_modules_layer1_modules_0_modules_conv2_modules_1_buffers_running_mean_
        l_self_modules_layer1_modules_0_modules_conv2_modules_1_buffers_running_var_ = (
            L_self_modules_layer1_modules_0_modules_conv2_modules_1_buffers_running_var_
        )
        l_self_modules_layer1_modules_0_modules_conv2_modules_1_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_conv2_modules_1_parameters_weight_
        )
        l_self_modules_layer1_modules_0_modules_conv2_modules_1_parameters_bias_ = (
            L_self_modules_layer1_modules_0_modules_conv2_modules_1_parameters_bias_
        )
        l_self_modules_layer1_modules_1_modules_conv1_modules_0_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_conv1_modules_0_parameters_weight_
        )
        l_self_modules_layer1_modules_1_modules_conv1_modules_1_buffers_running_mean_ = L_self_modules_layer1_modules_1_modules_conv1_modules_1_buffers_running_mean_
        l_self_modules_layer1_modules_1_modules_conv1_modules_1_buffers_running_var_ = (
            L_self_modules_layer1_modules_1_modules_conv1_modules_1_buffers_running_var_
        )
        l_self_modules_layer1_modules_1_modules_conv1_modules_1_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_conv1_modules_1_parameters_weight_
        )
        l_self_modules_layer1_modules_1_modules_conv1_modules_1_parameters_bias_ = (
            L_self_modules_layer1_modules_1_modules_conv1_modules_1_parameters_bias_
        )
        l_self_modules_layer1_modules_1_modules_conv2_modules_0_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_conv2_modules_0_parameters_weight_
        )
        l_self_modules_layer1_modules_1_modules_conv2_modules_1_buffers_running_mean_ = L_self_modules_layer1_modules_1_modules_conv2_modules_1_buffers_running_mean_
        l_self_modules_layer1_modules_1_modules_conv2_modules_1_buffers_running_var_ = (
            L_self_modules_layer1_modules_1_modules_conv2_modules_1_buffers_running_var_
        )
        l_self_modules_layer1_modules_1_modules_conv2_modules_1_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_conv2_modules_1_parameters_weight_
        )
        l_self_modules_layer1_modules_1_modules_conv2_modules_1_parameters_bias_ = (
            L_self_modules_layer1_modules_1_modules_conv2_modules_1_parameters_bias_
        )
        l_self_modules_layer2_modules_0_modules_conv1_modules_0_parameters_weight_ = (
            L_self_modules_layer2_modules_0_modules_conv1_modules_0_parameters_weight_
        )
        l_self_modules_layer2_modules_0_modules_conv1_modules_1_buffers_running_mean_ = L_self_modules_layer2_modules_0_modules_conv1_modules_1_buffers_running_mean_
        l_self_modules_layer2_modules_0_modules_conv1_modules_1_buffers_running_var_ = (
            L_self_modules_layer2_modules_0_modules_conv1_modules_1_buffers_running_var_
        )
        l_self_modules_layer2_modules_0_modules_conv1_modules_1_parameters_weight_ = (
            L_self_modules_layer2_modules_0_modules_conv1_modules_1_parameters_weight_
        )
        l_self_modules_layer2_modules_0_modules_conv1_modules_1_parameters_bias_ = (
            L_self_modules_layer2_modules_0_modules_conv1_modules_1_parameters_bias_
        )
        l_self_modules_layer2_modules_0_modules_conv2_modules_0_parameters_weight_ = (
            L_self_modules_layer2_modules_0_modules_conv2_modules_0_parameters_weight_
        )
        l_self_modules_layer2_modules_0_modules_conv2_modules_1_buffers_running_mean_ = L_self_modules_layer2_modules_0_modules_conv2_modules_1_buffers_running_mean_
        l_self_modules_layer2_modules_0_modules_conv2_modules_1_buffers_running_var_ = (
            L_self_modules_layer2_modules_0_modules_conv2_modules_1_buffers_running_var_
        )
        l_self_modules_layer2_modules_0_modules_conv2_modules_1_parameters_weight_ = (
            L_self_modules_layer2_modules_0_modules_conv2_modules_1_parameters_weight_
        )
        l_self_modules_layer2_modules_0_modules_conv2_modules_1_parameters_bias_ = (
            L_self_modules_layer2_modules_0_modules_conv2_modules_1_parameters_bias_
        )
        l_self_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_ = L_self_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_
        l_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_ = L_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_
        l_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_ = L_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_
        l_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_ = L_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_
        l_self_modules_layer2_modules_1_modules_conv1_modules_0_parameters_weight_ = (
            L_self_modules_layer2_modules_1_modules_conv1_modules_0_parameters_weight_
        )
        l_self_modules_layer2_modules_1_modules_conv1_modules_1_buffers_running_mean_ = L_self_modules_layer2_modules_1_modules_conv1_modules_1_buffers_running_mean_
        l_self_modules_layer2_modules_1_modules_conv1_modules_1_buffers_running_var_ = (
            L_self_modules_layer2_modules_1_modules_conv1_modules_1_buffers_running_var_
        )
        l_self_modules_layer2_modules_1_modules_conv1_modules_1_parameters_weight_ = (
            L_self_modules_layer2_modules_1_modules_conv1_modules_1_parameters_weight_
        )
        l_self_modules_layer2_modules_1_modules_conv1_modules_1_parameters_bias_ = (
            L_self_modules_layer2_modules_1_modules_conv1_modules_1_parameters_bias_
        )
        l_self_modules_layer2_modules_1_modules_conv2_modules_0_parameters_weight_ = (
            L_self_modules_layer2_modules_1_modules_conv2_modules_0_parameters_weight_
        )
        l_self_modules_layer2_modules_1_modules_conv2_modules_1_buffers_running_mean_ = L_self_modules_layer2_modules_1_modules_conv2_modules_1_buffers_running_mean_
        l_self_modules_layer2_modules_1_modules_conv2_modules_1_buffers_running_var_ = (
            L_self_modules_layer2_modules_1_modules_conv2_modules_1_buffers_running_var_
        )
        l_self_modules_layer2_modules_1_modules_conv2_modules_1_parameters_weight_ = (
            L_self_modules_layer2_modules_1_modules_conv2_modules_1_parameters_weight_
        )
        l_self_modules_layer2_modules_1_modules_conv2_modules_1_parameters_bias_ = (
            L_self_modules_layer2_modules_1_modules_conv2_modules_1_parameters_bias_
        )
        l_self_modules_layer3_modules_0_modules_conv1_modules_0_parameters_weight_ = (
            L_self_modules_layer3_modules_0_modules_conv1_modules_0_parameters_weight_
        )
        l_self_modules_layer3_modules_0_modules_conv1_modules_1_buffers_running_mean_ = L_self_modules_layer3_modules_0_modules_conv1_modules_1_buffers_running_mean_
        l_self_modules_layer3_modules_0_modules_conv1_modules_1_buffers_running_var_ = (
            L_self_modules_layer3_modules_0_modules_conv1_modules_1_buffers_running_var_
        )
        l_self_modules_layer3_modules_0_modules_conv1_modules_1_parameters_weight_ = (
            L_self_modules_layer3_modules_0_modules_conv1_modules_1_parameters_weight_
        )
        l_self_modules_layer3_modules_0_modules_conv1_modules_1_parameters_bias_ = (
            L_self_modules_layer3_modules_0_modules_conv1_modules_1_parameters_bias_
        )
        l_self_modules_layer3_modules_0_modules_conv2_modules_0_parameters_weight_ = (
            L_self_modules_layer3_modules_0_modules_conv2_modules_0_parameters_weight_
        )
        l_self_modules_layer3_modules_0_modules_conv2_modules_1_buffers_running_mean_ = L_self_modules_layer3_modules_0_modules_conv2_modules_1_buffers_running_mean_
        l_self_modules_layer3_modules_0_modules_conv2_modules_1_buffers_running_var_ = (
            L_self_modules_layer3_modules_0_modules_conv2_modules_1_buffers_running_var_
        )
        l_self_modules_layer3_modules_0_modules_conv2_modules_1_parameters_weight_ = (
            L_self_modules_layer3_modules_0_modules_conv2_modules_1_parameters_weight_
        )
        l_self_modules_layer3_modules_0_modules_conv2_modules_1_parameters_bias_ = (
            L_self_modules_layer3_modules_0_modules_conv2_modules_1_parameters_bias_
        )
        l_self_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_ = L_self_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_
        l_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_ = L_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_
        l_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_ = L_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_
        l_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_ = L_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_
        l_self_modules_layer3_modules_1_modules_conv1_modules_0_parameters_weight_ = (
            L_self_modules_layer3_modules_1_modules_conv1_modules_0_parameters_weight_
        )
        l_self_modules_layer3_modules_1_modules_conv1_modules_1_buffers_running_mean_ = L_self_modules_layer3_modules_1_modules_conv1_modules_1_buffers_running_mean_
        l_self_modules_layer3_modules_1_modules_conv1_modules_1_buffers_running_var_ = (
            L_self_modules_layer3_modules_1_modules_conv1_modules_1_buffers_running_var_
        )
        l_self_modules_layer3_modules_1_modules_conv1_modules_1_parameters_weight_ = (
            L_self_modules_layer3_modules_1_modules_conv1_modules_1_parameters_weight_
        )
        l_self_modules_layer3_modules_1_modules_conv1_modules_1_parameters_bias_ = (
            L_self_modules_layer3_modules_1_modules_conv1_modules_1_parameters_bias_
        )
        l_self_modules_layer3_modules_1_modules_conv2_modules_0_parameters_weight_ = (
            L_self_modules_layer3_modules_1_modules_conv2_modules_0_parameters_weight_
        )
        l_self_modules_layer3_modules_1_modules_conv2_modules_1_buffers_running_mean_ = L_self_modules_layer3_modules_1_modules_conv2_modules_1_buffers_running_mean_
        l_self_modules_layer3_modules_1_modules_conv2_modules_1_buffers_running_var_ = (
            L_self_modules_layer3_modules_1_modules_conv2_modules_1_buffers_running_var_
        )
        l_self_modules_layer3_modules_1_modules_conv2_modules_1_parameters_weight_ = (
            L_self_modules_layer3_modules_1_modules_conv2_modules_1_parameters_weight_
        )
        l_self_modules_layer3_modules_1_modules_conv2_modules_1_parameters_bias_ = (
            L_self_modules_layer3_modules_1_modules_conv2_modules_1_parameters_bias_
        )
        l_self_modules_layer4_modules_0_modules_conv1_modules_0_parameters_weight_ = (
            L_self_modules_layer4_modules_0_modules_conv1_modules_0_parameters_weight_
        )
        l_self_modules_layer4_modules_0_modules_conv1_modules_1_buffers_running_mean_ = L_self_modules_layer4_modules_0_modules_conv1_modules_1_buffers_running_mean_
        l_self_modules_layer4_modules_0_modules_conv1_modules_1_buffers_running_var_ = (
            L_self_modules_layer4_modules_0_modules_conv1_modules_1_buffers_running_var_
        )
        l_self_modules_layer4_modules_0_modules_conv1_modules_1_parameters_weight_ = (
            L_self_modules_layer4_modules_0_modules_conv1_modules_1_parameters_weight_
        )
        l_self_modules_layer4_modules_0_modules_conv1_modules_1_parameters_bias_ = (
            L_self_modules_layer4_modules_0_modules_conv1_modules_1_parameters_bias_
        )
        l_self_modules_layer4_modules_0_modules_conv2_modules_0_parameters_weight_ = (
            L_self_modules_layer4_modules_0_modules_conv2_modules_0_parameters_weight_
        )
        l_self_modules_layer4_modules_0_modules_conv2_modules_1_buffers_running_mean_ = L_self_modules_layer4_modules_0_modules_conv2_modules_1_buffers_running_mean_
        l_self_modules_layer4_modules_0_modules_conv2_modules_1_buffers_running_var_ = (
            L_self_modules_layer4_modules_0_modules_conv2_modules_1_buffers_running_var_
        )
        l_self_modules_layer4_modules_0_modules_conv2_modules_1_parameters_weight_ = (
            L_self_modules_layer4_modules_0_modules_conv2_modules_1_parameters_weight_
        )
        l_self_modules_layer4_modules_0_modules_conv2_modules_1_parameters_bias_ = (
            L_self_modules_layer4_modules_0_modules_conv2_modules_1_parameters_bias_
        )
        l_self_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_ = L_self_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_
        l_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_ = L_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_
        l_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_ = L_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_
        l_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_ = L_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_
        l_self_modules_layer4_modules_1_modules_conv1_modules_0_parameters_weight_ = (
            L_self_modules_layer4_modules_1_modules_conv1_modules_0_parameters_weight_
        )
        l_self_modules_layer4_modules_1_modules_conv1_modules_1_buffers_running_mean_ = L_self_modules_layer4_modules_1_modules_conv1_modules_1_buffers_running_mean_
        l_self_modules_layer4_modules_1_modules_conv1_modules_1_buffers_running_var_ = (
            L_self_modules_layer4_modules_1_modules_conv1_modules_1_buffers_running_var_
        )
        l_self_modules_layer4_modules_1_modules_conv1_modules_1_parameters_weight_ = (
            L_self_modules_layer4_modules_1_modules_conv1_modules_1_parameters_weight_
        )
        l_self_modules_layer4_modules_1_modules_conv1_modules_1_parameters_bias_ = (
            L_self_modules_layer4_modules_1_modules_conv1_modules_1_parameters_bias_
        )
        l_self_modules_layer4_modules_1_modules_conv2_modules_0_parameters_weight_ = (
            L_self_modules_layer4_modules_1_modules_conv2_modules_0_parameters_weight_
        )
        l_self_modules_layer4_modules_1_modules_conv2_modules_1_buffers_running_mean_ = L_self_modules_layer4_modules_1_modules_conv2_modules_1_buffers_running_mean_
        l_self_modules_layer4_modules_1_modules_conv2_modules_1_buffers_running_var_ = (
            L_self_modules_layer4_modules_1_modules_conv2_modules_1_buffers_running_var_
        )
        l_self_modules_layer4_modules_1_modules_conv2_modules_1_parameters_weight_ = (
            L_self_modules_layer4_modules_1_modules_conv2_modules_1_parameters_weight_
        )
        l_self_modules_layer4_modules_1_modules_conv2_modules_1_parameters_bias_ = (
            L_self_modules_layer4_modules_1_modules_conv2_modules_1_parameters_bias_
        )
        l_self_modules_fc_parameters_weight_ = L_self_modules_fc_parameters_weight_
        l_self_modules_fc_parameters_bias_ = L_self_modules_fc_parameters_bias_
        input_1 = torch.conv3d(
            l_x_,
            l_self_modules_stem_modules_0_parameters_weight_,
            None,
            (1, 2, 2),
            (1, 3, 3),
            (1, 1, 1),
            1,
        )
        l_x_ = l_self_modules_stem_modules_0_parameters_weight_ = None
        input_2 = torch.nn.functional.batch_norm(
            input_1,
            l_self_modules_stem_modules_1_buffers_running_mean_,
            l_self_modules_stem_modules_1_buffers_running_var_,
            l_self_modules_stem_modules_1_parameters_weight_,
            l_self_modules_stem_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_1 = (
            l_self_modules_stem_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_stem_modules_1_buffers_running_var_
        ) = (
            l_self_modules_stem_modules_1_parameters_weight_
        ) = l_self_modules_stem_modules_1_parameters_bias_ = None
        input_3 = torch.nn.functional.relu(input_2, inplace=True)
        input_2 = None
        input_4 = torch.conv3d(
            input_3,
            l_self_modules_layer1_modules_0_modules_conv1_modules_0_parameters_weight_,
            None,
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
            1,
        )
        l_self_modules_layer1_modules_0_modules_conv1_modules_0_parameters_weight_ = (
            None
        )
        input_5 = torch.nn.functional.batch_norm(
            input_4,
            l_self_modules_layer1_modules_0_modules_conv1_modules_1_buffers_running_mean_,
            l_self_modules_layer1_modules_0_modules_conv1_modules_1_buffers_running_var_,
            l_self_modules_layer1_modules_0_modules_conv1_modules_1_parameters_weight_,
            l_self_modules_layer1_modules_0_modules_conv1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_4 = l_self_modules_layer1_modules_0_modules_conv1_modules_1_buffers_running_mean_ = (
            l_self_modules_layer1_modules_0_modules_conv1_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_0_modules_conv1_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_0_modules_conv1_modules_1_parameters_bias_
        ) = None
        input_6 = torch.nn.functional.relu(input_5, inplace=True)
        input_5 = None
        input_7 = torch.conv3d(
            input_6,
            l_self_modules_layer1_modules_0_modules_conv2_modules_0_parameters_weight_,
            None,
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
            1,
        )
        input_6 = (
            l_self_modules_layer1_modules_0_modules_conv2_modules_0_parameters_weight_
        ) = None
        input_8 = torch.nn.functional.batch_norm(
            input_7,
            l_self_modules_layer1_modules_0_modules_conv2_modules_1_buffers_running_mean_,
            l_self_modules_layer1_modules_0_modules_conv2_modules_1_buffers_running_var_,
            l_self_modules_layer1_modules_0_modules_conv2_modules_1_parameters_weight_,
            l_self_modules_layer1_modules_0_modules_conv2_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_7 = l_self_modules_layer1_modules_0_modules_conv2_modules_1_buffers_running_mean_ = (
            l_self_modules_layer1_modules_0_modules_conv2_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_0_modules_conv2_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_0_modules_conv2_modules_1_parameters_bias_
        ) = None
        input_8 += input_3
        out = input_8
        input_8 = input_3 = None
        out_1 = torch.nn.functional.relu(out, inplace=True)
        out = None
        input_9 = torch.conv3d(
            out_1,
            l_self_modules_layer1_modules_1_modules_conv1_modules_0_parameters_weight_,
            None,
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
            1,
        )
        l_self_modules_layer1_modules_1_modules_conv1_modules_0_parameters_weight_ = (
            None
        )
        input_10 = torch.nn.functional.batch_norm(
            input_9,
            l_self_modules_layer1_modules_1_modules_conv1_modules_1_buffers_running_mean_,
            l_self_modules_layer1_modules_1_modules_conv1_modules_1_buffers_running_var_,
            l_self_modules_layer1_modules_1_modules_conv1_modules_1_parameters_weight_,
            l_self_modules_layer1_modules_1_modules_conv1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_9 = l_self_modules_layer1_modules_1_modules_conv1_modules_1_buffers_running_mean_ = (
            l_self_modules_layer1_modules_1_modules_conv1_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_1_modules_conv1_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_1_modules_conv1_modules_1_parameters_bias_
        ) = None
        input_11 = torch.nn.functional.relu(input_10, inplace=True)
        input_10 = None
        input_12 = torch.conv3d(
            input_11,
            l_self_modules_layer1_modules_1_modules_conv2_modules_0_parameters_weight_,
            None,
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
            1,
        )
        input_11 = (
            l_self_modules_layer1_modules_1_modules_conv2_modules_0_parameters_weight_
        ) = None
        input_13 = torch.nn.functional.batch_norm(
            input_12,
            l_self_modules_layer1_modules_1_modules_conv2_modules_1_buffers_running_mean_,
            l_self_modules_layer1_modules_1_modules_conv2_modules_1_buffers_running_var_,
            l_self_modules_layer1_modules_1_modules_conv2_modules_1_parameters_weight_,
            l_self_modules_layer1_modules_1_modules_conv2_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_12 = l_self_modules_layer1_modules_1_modules_conv2_modules_1_buffers_running_mean_ = (
            l_self_modules_layer1_modules_1_modules_conv2_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_1_modules_conv2_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_1_modules_conv2_modules_1_parameters_bias_
        ) = None
        input_13 += out_1
        out_2 = input_13
        input_13 = out_1 = None
        out_3 = torch.nn.functional.relu(out_2, inplace=True)
        out_2 = None
        input_14 = torch.conv3d(
            out_3,
            l_self_modules_layer2_modules_0_modules_conv1_modules_0_parameters_weight_,
            None,
            (2, 2, 2),
            (1, 1, 1),
            (1, 1, 1),
            1,
        )
        l_self_modules_layer2_modules_0_modules_conv1_modules_0_parameters_weight_ = (
            None
        )
        input_15 = torch.nn.functional.batch_norm(
            input_14,
            l_self_modules_layer2_modules_0_modules_conv1_modules_1_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_conv1_modules_1_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_conv1_modules_1_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_conv1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_14 = l_self_modules_layer2_modules_0_modules_conv1_modules_1_buffers_running_mean_ = (
            l_self_modules_layer2_modules_0_modules_conv1_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_0_modules_conv1_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_0_modules_conv1_modules_1_parameters_bias_
        ) = None
        input_16 = torch.nn.functional.relu(input_15, inplace=True)
        input_15 = None
        input_17 = torch.conv3d(
            input_16,
            l_self_modules_layer2_modules_0_modules_conv2_modules_0_parameters_weight_,
            None,
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
            1,
        )
        input_16 = (
            l_self_modules_layer2_modules_0_modules_conv2_modules_0_parameters_weight_
        ) = None
        input_18 = torch.nn.functional.batch_norm(
            input_17,
            l_self_modules_layer2_modules_0_modules_conv2_modules_1_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_conv2_modules_1_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_conv2_modules_1_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_conv2_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_17 = l_self_modules_layer2_modules_0_modules_conv2_modules_1_buffers_running_mean_ = (
            l_self_modules_layer2_modules_0_modules_conv2_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_0_modules_conv2_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_0_modules_conv2_modules_1_parameters_bias_
        ) = None
        input_19 = torch.conv3d(
            out_3,
            l_self_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (2, 2, 2),
            (0, 0, 0),
            (1, 1, 1),
            1,
        )
        out_3 = l_self_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_20 = torch.nn.functional.batch_norm(
            input_19,
            l_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_19 = l_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        input_18 += input_20
        out_4 = input_18
        input_18 = input_20 = None
        out_5 = torch.nn.functional.relu(out_4, inplace=True)
        out_4 = None
        input_21 = torch.conv3d(
            out_5,
            l_self_modules_layer2_modules_1_modules_conv1_modules_0_parameters_weight_,
            None,
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
            1,
        )
        l_self_modules_layer2_modules_1_modules_conv1_modules_0_parameters_weight_ = (
            None
        )
        input_22 = torch.nn.functional.batch_norm(
            input_21,
            l_self_modules_layer2_modules_1_modules_conv1_modules_1_buffers_running_mean_,
            l_self_modules_layer2_modules_1_modules_conv1_modules_1_buffers_running_var_,
            l_self_modules_layer2_modules_1_modules_conv1_modules_1_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_conv1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_21 = l_self_modules_layer2_modules_1_modules_conv1_modules_1_buffers_running_mean_ = (
            l_self_modules_layer2_modules_1_modules_conv1_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_1_modules_conv1_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_1_modules_conv1_modules_1_parameters_bias_
        ) = None
        input_23 = torch.nn.functional.relu(input_22, inplace=True)
        input_22 = None
        input_24 = torch.conv3d(
            input_23,
            l_self_modules_layer2_modules_1_modules_conv2_modules_0_parameters_weight_,
            None,
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
            1,
        )
        input_23 = (
            l_self_modules_layer2_modules_1_modules_conv2_modules_0_parameters_weight_
        ) = None
        input_25 = torch.nn.functional.batch_norm(
            input_24,
            l_self_modules_layer2_modules_1_modules_conv2_modules_1_buffers_running_mean_,
            l_self_modules_layer2_modules_1_modules_conv2_modules_1_buffers_running_var_,
            l_self_modules_layer2_modules_1_modules_conv2_modules_1_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_conv2_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_24 = l_self_modules_layer2_modules_1_modules_conv2_modules_1_buffers_running_mean_ = (
            l_self_modules_layer2_modules_1_modules_conv2_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_1_modules_conv2_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_1_modules_conv2_modules_1_parameters_bias_
        ) = None
        input_25 += out_5
        out_6 = input_25
        input_25 = out_5 = None
        out_7 = torch.nn.functional.relu(out_6, inplace=True)
        out_6 = None
        input_26 = torch.conv3d(
            out_7,
            l_self_modules_layer3_modules_0_modules_conv1_modules_0_parameters_weight_,
            None,
            (2, 2, 2),
            (1, 1, 1),
            (1, 1, 1),
            1,
        )
        l_self_modules_layer3_modules_0_modules_conv1_modules_0_parameters_weight_ = (
            None
        )
        input_27 = torch.nn.functional.batch_norm(
            input_26,
            l_self_modules_layer3_modules_0_modules_conv1_modules_1_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_conv1_modules_1_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_conv1_modules_1_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_conv1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_26 = l_self_modules_layer3_modules_0_modules_conv1_modules_1_buffers_running_mean_ = (
            l_self_modules_layer3_modules_0_modules_conv1_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_0_modules_conv1_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_0_modules_conv1_modules_1_parameters_bias_
        ) = None
        input_28 = torch.nn.functional.relu(input_27, inplace=True)
        input_27 = None
        input_29 = torch.conv3d(
            input_28,
            l_self_modules_layer3_modules_0_modules_conv2_modules_0_parameters_weight_,
            None,
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
            1,
        )
        input_28 = (
            l_self_modules_layer3_modules_0_modules_conv2_modules_0_parameters_weight_
        ) = None
        input_30 = torch.nn.functional.batch_norm(
            input_29,
            l_self_modules_layer3_modules_0_modules_conv2_modules_1_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_conv2_modules_1_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_conv2_modules_1_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_conv2_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_29 = l_self_modules_layer3_modules_0_modules_conv2_modules_1_buffers_running_mean_ = (
            l_self_modules_layer3_modules_0_modules_conv2_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_0_modules_conv2_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_0_modules_conv2_modules_1_parameters_bias_
        ) = None
        input_31 = torch.conv3d(
            out_7,
            l_self_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (2, 2, 2),
            (0, 0, 0),
            (1, 1, 1),
            1,
        )
        out_7 = l_self_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_32 = torch.nn.functional.batch_norm(
            input_31,
            l_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_31 = l_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        input_30 += input_32
        out_8 = input_30
        input_30 = input_32 = None
        out_9 = torch.nn.functional.relu(out_8, inplace=True)
        out_8 = None
        input_33 = torch.conv3d(
            out_9,
            l_self_modules_layer3_modules_1_modules_conv1_modules_0_parameters_weight_,
            None,
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
            1,
        )
        l_self_modules_layer3_modules_1_modules_conv1_modules_0_parameters_weight_ = (
            None
        )
        input_34 = torch.nn.functional.batch_norm(
            input_33,
            l_self_modules_layer3_modules_1_modules_conv1_modules_1_buffers_running_mean_,
            l_self_modules_layer3_modules_1_modules_conv1_modules_1_buffers_running_var_,
            l_self_modules_layer3_modules_1_modules_conv1_modules_1_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_conv1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_33 = l_self_modules_layer3_modules_1_modules_conv1_modules_1_buffers_running_mean_ = (
            l_self_modules_layer3_modules_1_modules_conv1_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_1_modules_conv1_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_1_modules_conv1_modules_1_parameters_bias_
        ) = None
        input_35 = torch.nn.functional.relu(input_34, inplace=True)
        input_34 = None
        input_36 = torch.conv3d(
            input_35,
            l_self_modules_layer3_modules_1_modules_conv2_modules_0_parameters_weight_,
            None,
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
            1,
        )
        input_35 = (
            l_self_modules_layer3_modules_1_modules_conv2_modules_0_parameters_weight_
        ) = None
        input_37 = torch.nn.functional.batch_norm(
            input_36,
            l_self_modules_layer3_modules_1_modules_conv2_modules_1_buffers_running_mean_,
            l_self_modules_layer3_modules_1_modules_conv2_modules_1_buffers_running_var_,
            l_self_modules_layer3_modules_1_modules_conv2_modules_1_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_conv2_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_36 = l_self_modules_layer3_modules_1_modules_conv2_modules_1_buffers_running_mean_ = (
            l_self_modules_layer3_modules_1_modules_conv2_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_1_modules_conv2_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_1_modules_conv2_modules_1_parameters_bias_
        ) = None
        input_37 += out_9
        out_10 = input_37
        input_37 = out_9 = None
        out_11 = torch.nn.functional.relu(out_10, inplace=True)
        out_10 = None
        input_38 = torch.conv3d(
            out_11,
            l_self_modules_layer4_modules_0_modules_conv1_modules_0_parameters_weight_,
            None,
            (2, 2, 2),
            (1, 1, 1),
            (1, 1, 1),
            1,
        )
        l_self_modules_layer4_modules_0_modules_conv1_modules_0_parameters_weight_ = (
            None
        )
        input_39 = torch.nn.functional.batch_norm(
            input_38,
            l_self_modules_layer4_modules_0_modules_conv1_modules_1_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_conv1_modules_1_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_conv1_modules_1_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_conv1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_38 = l_self_modules_layer4_modules_0_modules_conv1_modules_1_buffers_running_mean_ = (
            l_self_modules_layer4_modules_0_modules_conv1_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_0_modules_conv1_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_0_modules_conv1_modules_1_parameters_bias_
        ) = None
        input_40 = torch.nn.functional.relu(input_39, inplace=True)
        input_39 = None
        input_41 = torch.conv3d(
            input_40,
            l_self_modules_layer4_modules_0_modules_conv2_modules_0_parameters_weight_,
            None,
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
            1,
        )
        input_40 = (
            l_self_modules_layer4_modules_0_modules_conv2_modules_0_parameters_weight_
        ) = None
        input_42 = torch.nn.functional.batch_norm(
            input_41,
            l_self_modules_layer4_modules_0_modules_conv2_modules_1_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_conv2_modules_1_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_conv2_modules_1_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_conv2_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_41 = l_self_modules_layer4_modules_0_modules_conv2_modules_1_buffers_running_mean_ = (
            l_self_modules_layer4_modules_0_modules_conv2_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_0_modules_conv2_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_0_modules_conv2_modules_1_parameters_bias_
        ) = None
        input_43 = torch.conv3d(
            out_11,
            l_self_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (2, 2, 2),
            (0, 0, 0),
            (1, 1, 1),
            1,
        )
        out_11 = l_self_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_44 = torch.nn.functional.batch_norm(
            input_43,
            l_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_43 = l_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        input_42 += input_44
        out_12 = input_42
        input_42 = input_44 = None
        out_13 = torch.nn.functional.relu(out_12, inplace=True)
        out_12 = None
        input_45 = torch.conv3d(
            out_13,
            l_self_modules_layer4_modules_1_modules_conv1_modules_0_parameters_weight_,
            None,
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
            1,
        )
        l_self_modules_layer4_modules_1_modules_conv1_modules_0_parameters_weight_ = (
            None
        )
        input_46 = torch.nn.functional.batch_norm(
            input_45,
            l_self_modules_layer4_modules_1_modules_conv1_modules_1_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_conv1_modules_1_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_conv1_modules_1_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_conv1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_45 = l_self_modules_layer4_modules_1_modules_conv1_modules_1_buffers_running_mean_ = (
            l_self_modules_layer4_modules_1_modules_conv1_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_1_modules_conv1_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_1_modules_conv1_modules_1_parameters_bias_
        ) = None
        input_47 = torch.nn.functional.relu(input_46, inplace=True)
        input_46 = None
        input_48 = torch.conv3d(
            input_47,
            l_self_modules_layer4_modules_1_modules_conv2_modules_0_parameters_weight_,
            None,
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
            1,
        )
        input_47 = (
            l_self_modules_layer4_modules_1_modules_conv2_modules_0_parameters_weight_
        ) = None
        input_49 = torch.nn.functional.batch_norm(
            input_48,
            l_self_modules_layer4_modules_1_modules_conv2_modules_1_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_conv2_modules_1_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_conv2_modules_1_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_conv2_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_48 = l_self_modules_layer4_modules_1_modules_conv2_modules_1_buffers_running_mean_ = (
            l_self_modules_layer4_modules_1_modules_conv2_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_1_modules_conv2_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_1_modules_conv2_modules_1_parameters_bias_
        ) = None
        input_49 += out_13
        out_14 = input_49
        input_49 = out_13 = None
        out_15 = torch.nn.functional.relu(out_14, inplace=True)
        out_14 = None
        x = torch.nn.functional.adaptive_avg_pool3d(out_15, (1, 1, 1))
        out_15 = None
        x_1 = x.flatten(1)
        x = None
        x_2 = torch._C._nn.linear(
            x_1,
            l_self_modules_fc_parameters_weight_,
            l_self_modules_fc_parameters_bias_,
        )
        x_1 = (
            l_self_modules_fc_parameters_weight_
        ) = l_self_modules_fc_parameters_bias_ = None
        return (x_2,)
