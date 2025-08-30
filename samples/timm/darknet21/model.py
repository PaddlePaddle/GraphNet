import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_stem_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_x_: torch.Tensor,
        L_self_modules_stem_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stem_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stem_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_conv_down_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_conv_down_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_conv_down_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_conv_down_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_conv_down_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_conv_down_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_conv_down_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_conv_down_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_conv_down_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_conv_down_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_conv_down_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_conv_down_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_conv_down_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_conv_down_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_conv_down_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_conv_down_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_conv_down_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_conv_down_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_conv_down_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_conv_down_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_4_modules_conv_down_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_4_modules_conv_down_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_4_modules_conv_down_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_4_modules_conv_down_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_4_modules_conv_down_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_stem_modules_conv1_modules_conv_parameters_weight_ = (
            L_self_modules_stem_modules_conv1_modules_conv_parameters_weight_
        )
        l_x_ = L_x_
        l_self_modules_stem_modules_conv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_stem_modules_conv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_stem_modules_conv1_modules_bn_buffers_running_var_ = (
            L_self_modules_stem_modules_conv1_modules_bn_buffers_running_var_
        )
        l_self_modules_stem_modules_conv1_modules_bn_parameters_weight_ = (
            L_self_modules_stem_modules_conv1_modules_bn_parameters_weight_
        )
        l_self_modules_stem_modules_conv1_modules_bn_parameters_bias_ = (
            L_self_modules_stem_modules_conv1_modules_bn_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_conv_down_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_conv_down_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_conv_down_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_conv_down_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_conv_down_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_conv_down_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_conv_down_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_conv_down_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_conv_down_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_conv_down_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_conv_down_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_conv_down_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_conv_down_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_conv_down_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_conv_down_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_conv_down_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_conv_down_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_conv_down_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_conv_down_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_conv_down_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_conv_down_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_conv_down_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_conv_down_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_conv_down_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_conv_down_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_conv_down_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_conv_down_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_conv_down_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_conv_down_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_conv_down_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_conv_down_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_conv_down_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_conv_down_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_conv_down_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_conv_down_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_conv_down_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_conv_down_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_conv_down_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_conv_down_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_conv_down_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_4_modules_conv_down_modules_conv_parameters_weight_ = L_self_modules_stages_modules_4_modules_conv_down_modules_conv_parameters_weight_
        l_self_modules_stages_modules_4_modules_conv_down_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_4_modules_conv_down_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_4_modules_conv_down_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_4_modules_conv_down_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_4_modules_conv_down_modules_bn_parameters_weight_ = L_self_modules_stages_modules_4_modules_conv_down_modules_bn_parameters_weight_
        l_self_modules_stages_modules_4_modules_conv_down_modules_bn_parameters_bias_ = L_self_modules_stages_modules_4_modules_conv_down_modules_bn_parameters_bias_
        l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_head_modules_fc_parameters_weight_ = (
            L_self_modules_head_modules_fc_parameters_weight_
        )
        l_self_modules_head_modules_fc_parameters_bias_ = (
            L_self_modules_head_modules_fc_parameters_bias_
        )
        x = torch.conv2d(
            l_x_,
            l_self_modules_stem_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_x_ = l_self_modules_stem_modules_conv1_modules_conv_parameters_weight_ = None
        x_1 = torch.nn.functional.batch_norm(
            x,
            l_self_modules_stem_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stem_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stem_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stem_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x = (
            l_self_modules_stem_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_stem_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_stem_modules_conv1_modules_bn_parameters_weight_
        ) = l_self_modules_stem_modules_conv1_modules_bn_parameters_bias_ = None
        x_2 = torch.nn.functional.leaky_relu(x_1, 0.01, True)
        x_1 = None
        x_3 = torch.conv2d(
            x_2,
            l_self_modules_stages_modules_0_modules_conv_down_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_2 = l_self_modules_stages_modules_0_modules_conv_down_modules_conv_parameters_weight_ = (None)
        x_4 = torch.nn.functional.batch_norm(
            x_3,
            l_self_modules_stages_modules_0_modules_conv_down_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_conv_down_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_conv_down_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_conv_down_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_3 = l_self_modules_stages_modules_0_modules_conv_down_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_conv_down_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_conv_down_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_conv_down_modules_bn_parameters_bias_ = (None)
        x_5 = torch.nn.functional.leaky_relu(x_4, 0.01, True)
        x_4 = None
        x_6 = torch.conv2d(
            x_5,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_7 = torch.nn.functional.batch_norm(
            x_6,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_6 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_8 = torch.nn.functional.leaky_relu(x_7, 0.01, True)
        x_7 = None
        x_9 = torch.conv2d(
            x_8,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_8 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_10 = torch.nn.functional.batch_norm(
            x_9,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_9 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_11 = torch.nn.functional.leaky_relu(x_10, 0.01, True)
        x_10 = None
        x_12 = x_11 + x_5
        x_11 = x_5 = None
        x_13 = torch.conv2d(
            x_12,
            l_self_modules_stages_modules_1_modules_conv_down_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_12 = l_self_modules_stages_modules_1_modules_conv_down_modules_conv_parameters_weight_ = (None)
        x_14 = torch.nn.functional.batch_norm(
            x_13,
            l_self_modules_stages_modules_1_modules_conv_down_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_conv_down_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_conv_down_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_conv_down_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_13 = l_self_modules_stages_modules_1_modules_conv_down_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_conv_down_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_conv_down_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_conv_down_modules_bn_parameters_bias_ = (None)
        x_15 = torch.nn.functional.leaky_relu(x_14, 0.01, True)
        x_14 = None
        x_16 = torch.conv2d(
            x_15,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_17 = torch.nn.functional.batch_norm(
            x_16,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_16 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_18 = torch.nn.functional.leaky_relu(x_17, 0.01, True)
        x_17 = None
        x_19 = torch.conv2d(
            x_18,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_18 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_20 = torch.nn.functional.batch_norm(
            x_19,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_19 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_21 = torch.nn.functional.leaky_relu(x_20, 0.01, True)
        x_20 = None
        x_22 = x_21 + x_15
        x_21 = x_15 = None
        x_23 = torch.conv2d(
            x_22,
            l_self_modules_stages_modules_2_modules_conv_down_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_22 = l_self_modules_stages_modules_2_modules_conv_down_modules_conv_parameters_weight_ = (None)
        x_24 = torch.nn.functional.batch_norm(
            x_23,
            l_self_modules_stages_modules_2_modules_conv_down_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_conv_down_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_conv_down_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_conv_down_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_23 = l_self_modules_stages_modules_2_modules_conv_down_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_conv_down_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_conv_down_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_conv_down_modules_bn_parameters_bias_ = (None)
        x_25 = torch.nn.functional.leaky_relu(x_24, 0.01, True)
        x_24 = None
        x_26 = torch.conv2d(
            x_25,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_27 = torch.nn.functional.batch_norm(
            x_26,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_26 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_28 = torch.nn.functional.leaky_relu(x_27, 0.01, True)
        x_27 = None
        x_29 = torch.conv2d(
            x_28,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_28 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_30 = torch.nn.functional.batch_norm(
            x_29,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_29 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_31 = torch.nn.functional.leaky_relu(x_30, 0.01, True)
        x_30 = None
        x_32 = x_31 + x_25
        x_31 = x_25 = None
        x_33 = torch.conv2d(
            x_32,
            l_self_modules_stages_modules_3_modules_conv_down_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_32 = l_self_modules_stages_modules_3_modules_conv_down_modules_conv_parameters_weight_ = (None)
        x_34 = torch.nn.functional.batch_norm(
            x_33,
            l_self_modules_stages_modules_3_modules_conv_down_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_conv_down_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_conv_down_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_conv_down_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_33 = l_self_modules_stages_modules_3_modules_conv_down_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_conv_down_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_conv_down_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_conv_down_modules_bn_parameters_bias_ = (None)
        x_35 = torch.nn.functional.leaky_relu(x_34, 0.01, True)
        x_34 = None
        x_36 = torch.conv2d(
            x_35,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_37 = torch.nn.functional.batch_norm(
            x_36,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_36 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_38 = torch.nn.functional.leaky_relu(x_37, 0.01, True)
        x_37 = None
        x_39 = torch.conv2d(
            x_38,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_38 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_40 = torch.nn.functional.batch_norm(
            x_39,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_39 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_41 = torch.nn.functional.leaky_relu(x_40, 0.01, True)
        x_40 = None
        x_42 = x_41 + x_35
        x_41 = x_35 = None
        x_43 = torch.conv2d(
            x_42,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_44 = torch.nn.functional.batch_norm(
            x_43,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_43 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_45 = torch.nn.functional.leaky_relu(x_44, 0.01, True)
        x_44 = None
        x_46 = torch.conv2d(
            x_45,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_45 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_47 = torch.nn.functional.batch_norm(
            x_46,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_46 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_48 = torch.nn.functional.leaky_relu(x_47, 0.01, True)
        x_47 = None
        x_49 = x_48 + x_42
        x_48 = x_42 = None
        x_50 = torch.conv2d(
            x_49,
            l_self_modules_stages_modules_4_modules_conv_down_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_49 = l_self_modules_stages_modules_4_modules_conv_down_modules_conv_parameters_weight_ = (None)
        x_51 = torch.nn.functional.batch_norm(
            x_50,
            l_self_modules_stages_modules_4_modules_conv_down_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_4_modules_conv_down_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_4_modules_conv_down_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_4_modules_conv_down_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_50 = l_self_modules_stages_modules_4_modules_conv_down_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_4_modules_conv_down_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_4_modules_conv_down_modules_bn_parameters_weight_ = l_self_modules_stages_modules_4_modules_conv_down_modules_bn_parameters_bias_ = (None)
        x_52 = torch.nn.functional.leaky_relu(x_51, 0.01, True)
        x_51 = None
        x_53 = torch.conv2d(
            x_52,
            l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_54 = torch.nn.functional.batch_norm(
            x_53,
            l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_53 = l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_55 = torch.nn.functional.leaky_relu(x_54, 0.01, True)
        x_54 = None
        x_56 = torch.conv2d(
            x_55,
            l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_55 = l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_57 = torch.nn.functional.batch_norm(
            x_56,
            l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_56 = l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_58 = torch.nn.functional.leaky_relu(x_57, 0.01, True)
        x_57 = None
        x_59 = x_58 + x_52
        x_58 = x_52 = None
        x_60 = torch.conv2d(
            x_59,
            l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_61 = torch.nn.functional.batch_norm(
            x_60,
            l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_60 = l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_62 = torch.nn.functional.leaky_relu(x_61, 0.01, True)
        x_61 = None
        x_63 = torch.conv2d(
            x_62,
            l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_62 = l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_64 = torch.nn.functional.batch_norm(
            x_63,
            l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_63 = l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_65 = torch.nn.functional.leaky_relu(x_64, 0.01, True)
        x_64 = None
        x_66 = x_65 + x_59
        x_65 = x_59 = None
        x_67 = torch.nn.functional.adaptive_avg_pool2d(x_66, 1)
        x_66 = None
        x_68 = x_67.flatten(1, -1)
        x_67 = None
        x_69 = torch.nn.functional.dropout(x_68, 0.0, False, False)
        x_68 = None
        x_70 = torch._C._nn.linear(
            x_69,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_69 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_70,)
