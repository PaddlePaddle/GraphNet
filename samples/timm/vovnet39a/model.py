import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_stem_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        s53: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_stem_modules_0_modules_bn_momentum: torch.Tensor,
        L_self_modules_stem_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stem_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stem_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_0_modules_bn_eps: torch.Tensor,
        L_self_modules_stem_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_1_modules_bn_momentum: torch.Tensor,
        L_self_modules_stem_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stem_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stem_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_1_modules_bn_eps: torch.Tensor,
        L_self_modules_stem_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_2_modules_bn_momentum: torch.Tensor,
        L_self_modules_stem_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stem_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stem_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_2_modules_bn_eps: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_momentum: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_eps: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_momentum: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_eps: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_momentum: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_eps: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_momentum: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_eps: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_momentum: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_eps: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_momentum: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_eps: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_momentum: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_eps: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_momentum: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_eps: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_momentum: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_eps: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_momentum: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_eps: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_momentum: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_eps: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_momentum: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_eps: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_momentum: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_eps: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_momentum: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_eps: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_momentum: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_eps: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_momentum: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_eps: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_momentum: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_eps: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_momentum: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_eps: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_momentum: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_eps: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_momentum: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_eps: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_momentum: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_eps: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_momentum: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_eps: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_momentum: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_eps: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_bn_momentum: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_bn_eps: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_momentum: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_eps: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_momentum: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_eps: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_momentum: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_eps: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_momentum: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_eps: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_momentum: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_eps: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_momentum: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_eps: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_momentum: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_eps: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_momentum: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_eps: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_momentum: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_eps: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_momentum: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_eps: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_momentum: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_eps: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_bn_momentum: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_bn_eps: torch.Tensor,
        L_self_modules_head_modules_drop_p: torch.Tensor,
        L_self_modules_head_modules_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_stem_modules_0_modules_conv_parameters_weight_ = (
            L_self_modules_stem_modules_0_modules_conv_parameters_weight_
        )
        l_x_ = L_x_
        l_self_modules_stem_modules_0_modules_bn_momentum = (
            L_self_modules_stem_modules_0_modules_bn_momentum
        )
        l_self_modules_stem_modules_0_modules_bn_buffers_running_mean_ = (
            L_self_modules_stem_modules_0_modules_bn_buffers_running_mean_
        )
        l_self_modules_stem_modules_0_modules_bn_buffers_running_var_ = (
            L_self_modules_stem_modules_0_modules_bn_buffers_running_var_
        )
        l_self_modules_stem_modules_0_modules_bn_parameters_weight_ = (
            L_self_modules_stem_modules_0_modules_bn_parameters_weight_
        )
        l_self_modules_stem_modules_0_modules_bn_parameters_bias_ = (
            L_self_modules_stem_modules_0_modules_bn_parameters_bias_
        )
        l_self_modules_stem_modules_0_modules_bn_eps = (
            L_self_modules_stem_modules_0_modules_bn_eps
        )
        l_self_modules_stem_modules_1_modules_conv_parameters_weight_ = (
            L_self_modules_stem_modules_1_modules_conv_parameters_weight_
        )
        l_self_modules_stem_modules_1_modules_bn_momentum = (
            L_self_modules_stem_modules_1_modules_bn_momentum
        )
        l_self_modules_stem_modules_1_modules_bn_buffers_running_mean_ = (
            L_self_modules_stem_modules_1_modules_bn_buffers_running_mean_
        )
        l_self_modules_stem_modules_1_modules_bn_buffers_running_var_ = (
            L_self_modules_stem_modules_1_modules_bn_buffers_running_var_
        )
        l_self_modules_stem_modules_1_modules_bn_parameters_weight_ = (
            L_self_modules_stem_modules_1_modules_bn_parameters_weight_
        )
        l_self_modules_stem_modules_1_modules_bn_parameters_bias_ = (
            L_self_modules_stem_modules_1_modules_bn_parameters_bias_
        )
        l_self_modules_stem_modules_1_modules_bn_eps = (
            L_self_modules_stem_modules_1_modules_bn_eps
        )
        l_self_modules_stem_modules_2_modules_conv_parameters_weight_ = (
            L_self_modules_stem_modules_2_modules_conv_parameters_weight_
        )
        l_self_modules_stem_modules_2_modules_bn_momentum = (
            L_self_modules_stem_modules_2_modules_bn_momentum
        )
        l_self_modules_stem_modules_2_modules_bn_buffers_running_mean_ = (
            L_self_modules_stem_modules_2_modules_bn_buffers_running_mean_
        )
        l_self_modules_stem_modules_2_modules_bn_buffers_running_var_ = (
            L_self_modules_stem_modules_2_modules_bn_buffers_running_var_
        )
        l_self_modules_stem_modules_2_modules_bn_parameters_weight_ = (
            L_self_modules_stem_modules_2_modules_bn_parameters_weight_
        )
        l_self_modules_stem_modules_2_modules_bn_parameters_bias_ = (
            L_self_modules_stem_modules_2_modules_bn_parameters_bias_
        )
        l_self_modules_stem_modules_2_modules_bn_eps = (
            L_self_modules_stem_modules_2_modules_bn_eps
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_momentum = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_momentum
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_eps = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_eps
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_momentum = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_momentum
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_eps = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_eps
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_momentum = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_momentum
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_eps = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_eps
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_momentum = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_momentum
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_eps = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_eps
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_momentum = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_momentum
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_eps = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_eps
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_momentum = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_momentum
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_eps = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_eps
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_momentum = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_momentum
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_eps = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_eps
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_momentum = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_momentum
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_eps = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_eps
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_momentum = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_momentum
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_eps = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_eps
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_momentum = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_momentum
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_eps = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_eps
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_momentum = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_momentum
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_eps = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_eps
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_momentum = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_momentum
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_eps = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_eps
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_momentum = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_momentum
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_eps = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_eps
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_momentum = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_momentum
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_eps = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_eps
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_momentum = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_momentum
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_eps = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_eps
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_momentum = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_momentum
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_eps = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_eps
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_momentum = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_momentum
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_eps = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_eps
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_momentum = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_momentum
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_eps = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_eps
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_momentum = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_momentum
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_eps = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_eps
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_momentum = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_momentum
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_eps = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_eps
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_momentum = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_momentum
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_eps = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_eps
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_momentum = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_momentum
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_eps = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_eps
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_momentum = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_momentum
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_eps = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_eps
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_bn_momentum = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_bn_momentum
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_bn_eps = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_bn_eps
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_momentum = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_momentum
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_eps = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_eps
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_momentum = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_momentum
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_eps = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_eps
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_momentum = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_momentum
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_eps = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_eps
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_momentum = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_momentum
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_eps = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_eps
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_momentum = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_momentum
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_eps = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_eps
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_momentum = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_momentum
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_eps = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_eps
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_momentum = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_momentum
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_eps = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_eps
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_momentum = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_momentum
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_eps = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_eps
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_momentum = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_momentum
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_eps = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_eps
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_momentum = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_momentum
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_eps = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_eps
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_momentum = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_momentum
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_eps = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_eps
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_bn_momentum = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_bn_momentum
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_bn_eps = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_bn_eps
        l_self_modules_head_modules_drop_p = L_self_modules_head_modules_drop_p
        l_self_modules_head_modules_fc_parameters_weight_ = (
            L_self_modules_head_modules_fc_parameters_weight_
        )
        l_self_modules_head_modules_fc_parameters_bias_ = (
            L_self_modules_head_modules_fc_parameters_bias_
        )
        x = torch.conv2d(
            l_x_,
            l_self_modules_stem_modules_0_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_x_ = l_self_modules_stem_modules_0_modules_conv_parameters_weight_ = None
        item = l_self_modules_stem_modules_0_modules_bn_momentum.item()
        l_self_modules_stem_modules_0_modules_bn_momentum = None
        item_1 = l_self_modules_stem_modules_0_modules_bn_eps.item()
        l_self_modules_stem_modules_0_modules_bn_eps = None
        x_1 = torch.nn.functional.batch_norm(
            x,
            l_self_modules_stem_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stem_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stem_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stem_modules_0_modules_bn_parameters_bias_,
            False,
            item,
            item_1,
        )
        x = (
            l_self_modules_stem_modules_0_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_stem_modules_0_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_stem_modules_0_modules_bn_parameters_weight_
        ) = (
            l_self_modules_stem_modules_0_modules_bn_parameters_bias_
        ) = item = item_1 = None
        x_2 = torch.nn.functional.relu(x_1, inplace=True)
        x_1 = None
        x_3 = torch.conv2d(
            x_2,
            l_self_modules_stem_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_2 = l_self_modules_stem_modules_1_modules_conv_parameters_weight_ = None
        item_2 = l_self_modules_stem_modules_1_modules_bn_momentum.item()
        l_self_modules_stem_modules_1_modules_bn_momentum = None
        item_3 = l_self_modules_stem_modules_1_modules_bn_eps.item()
        l_self_modules_stem_modules_1_modules_bn_eps = None
        x_4 = torch.nn.functional.batch_norm(
            x_3,
            l_self_modules_stem_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stem_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stem_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stem_modules_1_modules_bn_parameters_bias_,
            False,
            item_2,
            item_3,
        )
        x_3 = (
            l_self_modules_stem_modules_1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_stem_modules_1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_stem_modules_1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_stem_modules_1_modules_bn_parameters_bias_
        ) = item_2 = item_3 = None
        x_5 = torch.nn.functional.relu(x_4, inplace=True)
        x_4 = None
        x_6 = torch.conv2d(
            x_5,
            l_self_modules_stem_modules_2_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_5 = l_self_modules_stem_modules_2_modules_conv_parameters_weight_ = None
        item_4 = l_self_modules_stem_modules_2_modules_bn_momentum.item()
        l_self_modules_stem_modules_2_modules_bn_momentum = None
        item_5 = l_self_modules_stem_modules_2_modules_bn_eps.item()
        l_self_modules_stem_modules_2_modules_bn_eps = None
        x_7 = torch.nn.functional.batch_norm(
            x_6,
            l_self_modules_stem_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stem_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stem_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stem_modules_2_modules_bn_parameters_bias_,
            False,
            item_4,
            item_5,
        )
        x_6 = (
            l_self_modules_stem_modules_2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_stem_modules_2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_stem_modules_2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_stem_modules_2_modules_bn_parameters_bias_
        ) = item_4 = item_5 = None
        x_8 = torch.nn.functional.relu(x_7, inplace=True)
        x_7 = None
        x_9 = torch.conv2d(
            x_8,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        item_6 = (
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_momentum.item()
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_momentum = (
            None
        )
        item_7 = (
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_eps.item()
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_eps = (
            None
        )
        x_10 = torch.nn.functional.batch_norm(
            x_9,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_bias_,
            False,
            item_6,
            item_7,
        )
        x_9 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_bias_ = (item_6) = (
            item_7
        ) = None
        x_11 = torch.nn.functional.relu(x_10, inplace=True)
        x_10 = None
        x_12 = torch.conv2d(
            x_11,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        item_8 = (
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_momentum.item()
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_momentum = (
            None
        )
        item_9 = (
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_eps.item()
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_eps = (
            None
        )
        x_13 = torch.nn.functional.batch_norm(
            x_12,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_bias_,
            False,
            item_8,
            item_9,
        )
        x_12 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_bias_ = (item_8) = (
            item_9
        ) = None
        x_14 = torch.nn.functional.relu(x_13, inplace=True)
        x_13 = None
        x_15 = torch.conv2d(
            x_14,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        item_10 = (
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_momentum.item()
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_momentum = (
            None
        )
        item_11 = (
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_eps.item()
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_eps = (
            None
        )
        x_16 = torch.nn.functional.batch_norm(
            x_15,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_bias_,
            False,
            item_10,
            item_11,
        )
        x_15 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_bias_ = (item_10) = (
            item_11
        ) = None
        x_17 = torch.nn.functional.relu(x_16, inplace=True)
        x_16 = None
        x_18 = torch.conv2d(
            x_17,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_conv_parameters_weight_ = (
            None
        )
        item_12 = (
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_momentum.item()
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_momentum = (
            None
        )
        item_13 = (
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_eps.item()
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_eps = (
            None
        )
        x_19 = torch.nn.functional.batch_norm(
            x_18,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_bias_,
            False,
            item_12,
            item_13,
        )
        x_18 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_bias_ = (item_12) = (
            item_13
        ) = None
        x_20 = torch.nn.functional.relu(x_19, inplace=True)
        x_19 = None
        x_21 = torch.conv2d(
            x_20,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_conv_parameters_weight_ = (
            None
        )
        item_14 = (
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_momentum.item()
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_momentum = (
            None
        )
        item_15 = (
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_eps.item()
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_eps = (
            None
        )
        x_22 = torch.nn.functional.batch_norm(
            x_21,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_bias_,
            False,
            item_14,
            item_15,
        )
        x_21 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_bias_ = (item_14) = (
            item_15
        ) = None
        x_23 = torch.nn.functional.relu(x_22, inplace=True)
        x_22 = None
        x_24 = torch.cat([x_8, x_11, x_14, x_17, x_20, x_23], dim=1)
        x_8 = x_11 = x_14 = x_17 = x_20 = x_23 = None
        x_25 = torch.conv2d(
            x_24,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_24 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_conv_parameters_weight_ = (None)
        item_16 = (
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_momentum.item()
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_momentum = (
            None
        )
        item_17 = (
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_eps.item()
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_eps = (
            None
        )
        x_26 = torch.nn.functional.batch_norm(
            x_25,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_bias_,
            False,
            item_16,
            item_17,
        )
        x_25 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_bias_ = (item_16) = (
            item_17
        ) = None
        x_27 = torch.nn.functional.relu(x_26, inplace=True)
        x_26 = None
        x_28 = torch.nn.functional.max_pool2d(
            x_27, 3, 2, 0, 1, ceil_mode=True, return_indices=False
        )
        x_27 = None
        x_29 = torch.conv2d(
            x_28,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        item_18 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_momentum.item()
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_momentum = (
            None
        )
        item_19 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_eps.item()
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_eps = (
            None
        )
        x_30 = torch.nn.functional.batch_norm(
            x_29,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_bias_,
            False,
            item_18,
            item_19,
        )
        x_29 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_bias_ = (item_18) = (
            item_19
        ) = None
        x_31 = torch.nn.functional.relu(x_30, inplace=True)
        x_30 = None
        x_32 = torch.conv2d(
            x_31,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        item_20 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_momentum.item()
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_momentum = (
            None
        )
        item_21 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_eps.item()
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_eps = (
            None
        )
        x_33 = torch.nn.functional.batch_norm(
            x_32,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_bias_,
            False,
            item_20,
            item_21,
        )
        x_32 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_bias_ = (item_20) = (
            item_21
        ) = None
        x_34 = torch.nn.functional.relu(x_33, inplace=True)
        x_33 = None
        x_35 = torch.conv2d(
            x_34,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        item_22 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_momentum.item()
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_momentum = (
            None
        )
        item_23 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_eps.item()
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_eps = (
            None
        )
        x_36 = torch.nn.functional.batch_norm(
            x_35,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_bias_,
            False,
            item_22,
            item_23,
        )
        x_35 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_bias_ = (item_22) = (
            item_23
        ) = None
        x_37 = torch.nn.functional.relu(x_36, inplace=True)
        x_36 = None
        x_38 = torch.conv2d(
            x_37,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_conv_parameters_weight_ = (
            None
        )
        item_24 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_momentum.item()
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_momentum = (
            None
        )
        item_25 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_eps.item()
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_eps = (
            None
        )
        x_39 = torch.nn.functional.batch_norm(
            x_38,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_bias_,
            False,
            item_24,
            item_25,
        )
        x_38 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_bias_ = (item_24) = (
            item_25
        ) = None
        x_40 = torch.nn.functional.relu(x_39, inplace=True)
        x_39 = None
        x_41 = torch.conv2d(
            x_40,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_conv_parameters_weight_ = (
            None
        )
        item_26 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_momentum.item()
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_momentum = (
            None
        )
        item_27 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_eps.item()
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_eps = (
            None
        )
        x_42 = torch.nn.functional.batch_norm(
            x_41,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_bias_,
            False,
            item_26,
            item_27,
        )
        x_41 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_bias_ = (item_26) = (
            item_27
        ) = None
        x_43 = torch.nn.functional.relu(x_42, inplace=True)
        x_42 = None
        x_44 = torch.cat([x_28, x_31, x_34, x_37, x_40, x_43], dim=1)
        x_28 = x_31 = x_34 = x_37 = x_40 = x_43 = None
        x_45 = torch.conv2d(
            x_44,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_44 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_conv_parameters_weight_ = (None)
        item_28 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_momentum.item()
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_momentum = (
            None
        )
        item_29 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_eps.item()
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_eps = (
            None
        )
        x_46 = torch.nn.functional.batch_norm(
            x_45,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_bias_,
            False,
            item_28,
            item_29,
        )
        x_45 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_bias_ = (item_28) = (
            item_29
        ) = None
        x_47 = torch.nn.functional.relu(x_46, inplace=True)
        x_46 = None
        x_48 = torch.nn.functional.max_pool2d(
            x_47, 3, 2, 0, 1, ceil_mode=True, return_indices=False
        )
        x_47 = None
        x_49 = torch.conv2d(
            x_48,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        item_30 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_momentum.item()
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_momentum = (
            None
        )
        item_31 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_eps.item()
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_eps = (
            None
        )
        x_50 = torch.nn.functional.batch_norm(
            x_49,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_bias_,
            False,
            item_30,
            item_31,
        )
        x_49 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_bias_ = (item_30) = (
            item_31
        ) = None
        x_51 = torch.nn.functional.relu(x_50, inplace=True)
        x_50 = None
        x_52 = torch.conv2d(
            x_51,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        item_32 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_momentum.item()
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_momentum = (
            None
        )
        item_33 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_eps.item()
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_eps = (
            None
        )
        x_53 = torch.nn.functional.batch_norm(
            x_52,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_bias_,
            False,
            item_32,
            item_33,
        )
        x_52 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_bias_ = (item_32) = (
            item_33
        ) = None
        x_54 = torch.nn.functional.relu(x_53, inplace=True)
        x_53 = None
        x_55 = torch.conv2d(
            x_54,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        item_34 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_momentum.item()
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_momentum = (
            None
        )
        item_35 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_eps.item()
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_eps = (
            None
        )
        x_56 = torch.nn.functional.batch_norm(
            x_55,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_bias_,
            False,
            item_34,
            item_35,
        )
        x_55 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_bias_ = (item_34) = (
            item_35
        ) = None
        x_57 = torch.nn.functional.relu(x_56, inplace=True)
        x_56 = None
        x_58 = torch.conv2d(
            x_57,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_conv_parameters_weight_ = (
            None
        )
        item_36 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_momentum.item()
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_momentum = (
            None
        )
        item_37 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_eps.item()
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_eps = (
            None
        )
        x_59 = torch.nn.functional.batch_norm(
            x_58,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_bias_,
            False,
            item_36,
            item_37,
        )
        x_58 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_bias_ = (item_36) = (
            item_37
        ) = None
        x_60 = torch.nn.functional.relu(x_59, inplace=True)
        x_59 = None
        x_61 = torch.conv2d(
            x_60,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_conv_parameters_weight_ = (
            None
        )
        item_38 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_momentum.item()
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_momentum = (
            None
        )
        item_39 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_eps.item()
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_eps = (
            None
        )
        x_62 = torch.nn.functional.batch_norm(
            x_61,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_bias_,
            False,
            item_38,
            item_39,
        )
        x_61 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_bias_ = (item_38) = (
            item_39
        ) = None
        x_63 = torch.nn.functional.relu(x_62, inplace=True)
        x_62 = None
        x_64 = torch.cat([x_48, x_51, x_54, x_57, x_60, x_63], dim=1)
        x_48 = x_51 = x_54 = x_57 = x_60 = x_63 = None
        x_65 = torch.conv2d(
            x_64,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_64 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_conv_parameters_weight_ = (None)
        item_40 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_momentum.item()
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_momentum = (
            None
        )
        item_41 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_eps.item()
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_eps = (
            None
        )
        x_66 = torch.nn.functional.batch_norm(
            x_65,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_bias_,
            False,
            item_40,
            item_41,
        )
        x_65 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_bias_ = (item_40) = (
            item_41
        ) = None
        x_67 = torch.nn.functional.relu(x_66, inplace=True)
        x_66 = None
        x_68 = torch.conv2d(
            x_67,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        item_42 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_momentum.item()
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_momentum = (
            None
        )
        item_43 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_eps.item()
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_eps = (
            None
        )
        x_69 = torch.nn.functional.batch_norm(
            x_68,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_parameters_bias_,
            False,
            item_42,
            item_43,
        )
        x_68 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_parameters_bias_ = (item_42) = (
            item_43
        ) = None
        x_70 = torch.nn.functional.relu(x_69, inplace=True)
        x_69 = None
        x_71 = torch.conv2d(
            x_70,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        item_44 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_momentum.item()
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_momentum = (
            None
        )
        item_45 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_eps.item()
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_eps = (
            None
        )
        x_72 = torch.nn.functional.batch_norm(
            x_71,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_parameters_bias_,
            False,
            item_44,
            item_45,
        )
        x_71 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_parameters_bias_ = (item_44) = (
            item_45
        ) = None
        x_73 = torch.nn.functional.relu(x_72, inplace=True)
        x_72 = None
        x_74 = torch.conv2d(
            x_73,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        item_46 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_momentum.item()
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_momentum = (
            None
        )
        item_47 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_eps.item()
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_eps = (
            None
        )
        x_75 = torch.nn.functional.batch_norm(
            x_74,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_parameters_bias_,
            False,
            item_46,
            item_47,
        )
        x_74 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_parameters_bias_ = (item_46) = (
            item_47
        ) = None
        x_76 = torch.nn.functional.relu(x_75, inplace=True)
        x_75 = None
        x_77 = torch.conv2d(
            x_76,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_conv_parameters_weight_ = (
            None
        )
        item_48 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_momentum.item()
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_momentum = (
            None
        )
        item_49 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_eps.item()
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_eps = (
            None
        )
        x_78 = torch.nn.functional.batch_norm(
            x_77,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_parameters_bias_,
            False,
            item_48,
            item_49,
        )
        x_77 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_parameters_bias_ = (item_48) = (
            item_49
        ) = None
        x_79 = torch.nn.functional.relu(x_78, inplace=True)
        x_78 = None
        x_80 = torch.conv2d(
            x_79,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_conv_parameters_weight_ = (
            None
        )
        item_50 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_momentum.item()
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_momentum = (
            None
        )
        item_51 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_eps.item()
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_eps = (
            None
        )
        x_81 = torch.nn.functional.batch_norm(
            x_80,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_parameters_bias_,
            False,
            item_50,
            item_51,
        )
        x_80 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_parameters_bias_ = (item_50) = (
            item_51
        ) = None
        x_82 = torch.nn.functional.relu(x_81, inplace=True)
        x_81 = None
        x_83 = torch.cat([x_67, x_70, x_73, x_76, x_79, x_82], dim=1)
        x_67 = x_70 = x_73 = x_76 = x_79 = x_82 = None
        x_84 = torch.conv2d(
            x_83,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_83 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_conv_parameters_weight_ = (None)
        item_52 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_bn_momentum.item()
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_bn_momentum = (
            None
        )
        item_53 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_bn_eps.item()
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_bn_eps = (
            None
        )
        x_85 = torch.nn.functional.batch_norm(
            x_84,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_bn_parameters_bias_,
            False,
            item_52,
            item_53,
        )
        x_84 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_concat_modules_bn_parameters_bias_ = (item_52) = (
            item_53
        ) = None
        x_86 = torch.nn.functional.relu(x_85, inplace=True)
        x_85 = None
        x_87 = torch.nn.functional.max_pool2d(
            x_86, 3, 2, 0, 1, ceil_mode=True, return_indices=False
        )
        x_86 = None
        x_88 = torch.conv2d(
            x_87,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        item_54 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_momentum.item()
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_momentum = (
            None
        )
        item_55 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_eps.item()
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_eps = (
            None
        )
        x_89 = torch.nn.functional.batch_norm(
            x_88,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_bias_,
            False,
            item_54,
            item_55,
        )
        x_88 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_bias_ = (item_54) = (
            item_55
        ) = None
        x_90 = torch.nn.functional.relu(x_89, inplace=True)
        x_89 = None
        x_91 = torch.conv2d(
            x_90,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        item_56 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_momentum.item()
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_momentum = (
            None
        )
        item_57 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_eps.item()
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_eps = (
            None
        )
        x_92 = torch.nn.functional.batch_norm(
            x_91,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_bias_,
            False,
            item_56,
            item_57,
        )
        x_91 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_bias_ = (item_56) = (
            item_57
        ) = None
        x_93 = torch.nn.functional.relu(x_92, inplace=True)
        x_92 = None
        x_94 = torch.conv2d(
            x_93,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        item_58 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_momentum.item()
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_momentum = (
            None
        )
        item_59 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_eps.item()
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_eps = (
            None
        )
        x_95 = torch.nn.functional.batch_norm(
            x_94,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_bias_,
            False,
            item_58,
            item_59,
        )
        x_94 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_bias_ = (item_58) = (
            item_59
        ) = None
        x_96 = torch.nn.functional.relu(x_95, inplace=True)
        x_95 = None
        x_97 = torch.conv2d(
            x_96,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_conv_parameters_weight_ = (
            None
        )
        item_60 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_momentum.item()
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_momentum = (
            None
        )
        item_61 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_eps.item()
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_eps = (
            None
        )
        x_98 = torch.nn.functional.batch_norm(
            x_97,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_bias_,
            False,
            item_60,
            item_61,
        )
        x_97 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_3_modules_bn_parameters_bias_ = (item_60) = (
            item_61
        ) = None
        x_99 = torch.nn.functional.relu(x_98, inplace=True)
        x_98 = None
        x_100 = torch.conv2d(
            x_99,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_conv_parameters_weight_ = (
            None
        )
        item_62 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_momentum.item()
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_momentum = (
            None
        )
        item_63 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_eps.item()
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_eps = (
            None
        )
        x_101 = torch.nn.functional.batch_norm(
            x_100,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_bias_,
            False,
            item_62,
            item_63,
        )
        x_100 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_4_modules_bn_parameters_bias_ = (item_62) = (
            item_63
        ) = None
        x_102 = torch.nn.functional.relu(x_101, inplace=True)
        x_101 = None
        x_103 = torch.cat([x_87, x_90, x_93, x_96, x_99, x_102], dim=1)
        x_87 = x_90 = x_93 = x_96 = x_99 = x_102 = None
        x_104 = torch.conv2d(
            x_103,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_103 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_conv_parameters_weight_ = (None)
        item_64 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_momentum.item()
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_momentum = (
            None
        )
        item_65 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_eps.item()
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_eps = (
            None
        )
        x_105 = torch.nn.functional.batch_norm(
            x_104,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_bias_,
            False,
            item_64,
            item_65,
        )
        x_104 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_bias_ = (item_64) = (
            item_65
        ) = None
        x_106 = torch.nn.functional.relu(x_105, inplace=True)
        x_105 = None
        x_107 = torch.conv2d(
            x_106,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        item_66 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_momentum.item()
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_momentum = (
            None
        )
        item_67 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_eps.item()
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_eps = (
            None
        )
        x_108 = torch.nn.functional.batch_norm(
            x_107,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_parameters_bias_,
            False,
            item_66,
            item_67,
        )
        x_107 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_0_modules_bn_parameters_bias_ = (item_66) = (
            item_67
        ) = None
        x_109 = torch.nn.functional.relu(x_108, inplace=True)
        x_108 = None
        x_110 = torch.conv2d(
            x_109,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        item_68 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_momentum.item()
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_momentum = (
            None
        )
        item_69 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_eps.item()
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_eps = (
            None
        )
        x_111 = torch.nn.functional.batch_norm(
            x_110,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_parameters_bias_,
            False,
            item_68,
            item_69,
        )
        x_110 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_1_modules_bn_parameters_bias_ = (item_68) = (
            item_69
        ) = None
        x_112 = torch.nn.functional.relu(x_111, inplace=True)
        x_111 = None
        x_113 = torch.conv2d(
            x_112,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        item_70 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_momentum.item()
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_momentum = (
            None
        )
        item_71 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_eps.item()
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_eps = (
            None
        )
        x_114 = torch.nn.functional.batch_norm(
            x_113,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_parameters_bias_,
            False,
            item_70,
            item_71,
        )
        x_113 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_2_modules_bn_parameters_bias_ = (item_70) = (
            item_71
        ) = None
        x_115 = torch.nn.functional.relu(x_114, inplace=True)
        x_114 = None
        x_116 = torch.conv2d(
            x_115,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_conv_parameters_weight_ = (
            None
        )
        item_72 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_momentum.item()
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_momentum = (
            None
        )
        item_73 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_eps.item()
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_eps = (
            None
        )
        x_117 = torch.nn.functional.batch_norm(
            x_116,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_parameters_bias_,
            False,
            item_72,
            item_73,
        )
        x_116 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_3_modules_bn_parameters_bias_ = (item_72) = (
            item_73
        ) = None
        x_118 = torch.nn.functional.relu(x_117, inplace=True)
        x_117 = None
        x_119 = torch.conv2d(
            x_118,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_conv_parameters_weight_ = (
            None
        )
        item_74 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_momentum.item()
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_momentum = (
            None
        )
        item_75 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_eps.item()
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_eps = (
            None
        )
        x_120 = torch.nn.functional.batch_norm(
            x_119,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_parameters_bias_,
            False,
            item_74,
            item_75,
        )
        x_119 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_mid_modules_4_modules_bn_parameters_bias_ = (item_74) = (
            item_75
        ) = None
        x_121 = torch.nn.functional.relu(x_120, inplace=True)
        x_120 = None
        x_122 = torch.cat([x_106, x_109, x_112, x_115, x_118, x_121], dim=1)
        x_106 = x_109 = x_112 = x_115 = x_118 = x_121 = None
        x_123 = torch.conv2d(
            x_122,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_122 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_conv_parameters_weight_ = (None)
        item_76 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_bn_momentum.item()
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_bn_momentum = (
            None
        )
        item_77 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_bn_eps.item()
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_bn_eps = (
            None
        )
        x_124 = torch.nn.functional.batch_norm(
            x_123,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_bn_parameters_bias_,
            False,
            item_76,
            item_77,
        )
        x_123 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_concat_modules_bn_parameters_bias_ = (item_76) = (
            item_77
        ) = None
        x_125 = torch.nn.functional.relu(x_124, inplace=True)
        x_124 = None
        x_126 = torch.nn.functional.adaptive_avg_pool2d(x_125, 1)
        x_125 = None
        x_127 = x_126.flatten(1, -1)
        x_126 = None
        item_78 = l_self_modules_head_modules_drop_p.item()
        l_self_modules_head_modules_drop_p = None
        x_128 = torch.nn.functional.dropout(x_127, item_78, False, False)
        x_127 = item_78 = None
        x_129 = torch.nn.functional.linear(
            x_128,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_128 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_129,)
