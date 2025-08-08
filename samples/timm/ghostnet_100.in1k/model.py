import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_conv_stem_parameters_weight_: torch.nn.parameter.Parameter,
        s1: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_0_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_0_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_0_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_0_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_3_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_3_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_3_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_3_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_4_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_4_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_4_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_4_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_0_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_0_modules_bn_dw_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_7_modules_0_modules_bn_dw_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_7_modules_0_modules_bn_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_0_modules_bn_dw_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_0_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_0_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_0_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_0_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_1_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_1_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_1_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_1_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_3_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_3_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_3_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_3_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_9_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_9_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_conv_head_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv_head_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_conv_stem_parameters_weight_ = (
            L_self_modules_conv_stem_parameters_weight_
        )
        l_x_ = L_x_
        l_self_modules_bn1_buffers_running_mean_ = (
            L_self_modules_bn1_buffers_running_mean_
        )
        l_self_modules_bn1_buffers_running_var_ = (
            L_self_modules_bn1_buffers_running_var_
        )
        l_self_modules_bn1_parameters_weight_ = L_self_modules_bn1_parameters_weight_
        l_self_modules_bn1_parameters_bias_ = L_self_modules_bn1_parameters_bias_
        l_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_ = L_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_
        l_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_ = L_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_
        l_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_ = L_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_
        l_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_ = L_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_
        l_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_ = L_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_
        l_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_ = L_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_
        l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_ = L_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_
        l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_ = L_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_
        l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_ = L_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_
        l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_ = L_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_
        l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_ = L_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_
        l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_ = L_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_
        l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_ = L_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_
        l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_ = L_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_
        l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_ = L_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_
        l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_ = L_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_
        l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_ = L_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_
        l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_ = L_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_
        l_self_modules_blocks_modules_1_modules_0_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_0_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_0_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_1_modules_0_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_1_modules_0_modules_bn_dw_buffers_running_var_ = (
            L_self_modules_blocks_modules_1_modules_0_modules_bn_dw_buffers_running_var_
        )
        l_self_modules_blocks_modules_1_modules_0_modules_bn_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_0_modules_bn_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_0_modules_bn_dw_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_0_modules_bn_dw_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_ = L_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_
        l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_ = L_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_
        l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_ = L_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_
        l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_ = L_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_
        l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_ = L_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_
        l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_ = L_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_
        l_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_0_parameters_weight_ = L_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_0_parameters_weight_
        l_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_1_parameters_weight_ = L_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_1_parameters_weight_
        l_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_1_parameters_bias_ = L_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_1_parameters_bias_
        l_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_2_parameters_weight_ = L_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_2_parameters_weight_
        l_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_3_buffers_running_mean_ = L_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_3_buffers_running_mean_
        l_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_3_buffers_running_var_ = L_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_3_buffers_running_var_
        l_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_3_parameters_weight_ = L_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_3_parameters_weight_
        l_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_3_parameters_bias_ = L_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_3_parameters_bias_
        l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_ = L_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_
        l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_ = L_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_
        l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_ = L_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_
        l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_ = L_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_
        l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_ = L_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_
        l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_ = L_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_
        l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_ = L_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_
        l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_ = L_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_
        l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_ = L_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_
        l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_ = L_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_
        l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_ = L_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_
        l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_ = L_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_
        l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_ = L_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_
        l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_ = L_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_
        l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_ = L_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_
        l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_ = L_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_
        l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_ = L_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_
        l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_ = L_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_
        l_self_modules_blocks_modules_3_modules_0_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_0_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_0_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_0_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_0_modules_bn_dw_buffers_running_var_ = (
            L_self_modules_blocks_modules_3_modules_0_modules_bn_dw_buffers_running_var_
        )
        l_self_modules_blocks_modules_3_modules_0_modules_bn_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_0_modules_bn_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_0_modules_bn_dw_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_0_modules_bn_dw_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_ = L_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_
        l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_ = L_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_
        l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_ = L_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_
        l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_ = L_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_
        l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_ = L_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_
        l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_ = L_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_
        l_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_0_parameters_weight_ = L_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_0_parameters_weight_
        l_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_1_parameters_weight_ = L_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_1_parameters_weight_
        l_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_1_parameters_bias_ = L_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_1_parameters_bias_
        l_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_2_parameters_weight_ = L_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_2_parameters_weight_
        l_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_3_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_3_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_3_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_3_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_3_parameters_weight_ = L_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_3_parameters_weight_
        l_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_3_parameters_bias_ = L_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_3_parameters_bias_
        l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_ = L_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_
        l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_ = L_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_
        l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_ = L_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_
        l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_ = L_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_
        l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_ = L_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_
        l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_ = L_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_
        l_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_ = L_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_
        l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_ = L_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_
        l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_ = L_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_
        l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_ = L_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_
        l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_ = L_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_
        l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_ = L_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_
        l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_ = L_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_
        l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_ = L_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_
        l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_ = L_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_
        l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_ = L_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_
        l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_ = L_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_
        l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_ = L_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_
        l_self_modules_blocks_modules_5_modules_0_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_0_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_0_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_5_modules_0_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_5_modules_0_modules_bn_dw_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_0_modules_bn_dw_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_0_modules_bn_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_0_modules_bn_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_0_modules_bn_dw_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_0_modules_bn_dw_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_ = L_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_
        l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_ = L_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_
        l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_ = L_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_
        l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_ = L_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_
        l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_ = L_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_
        l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_ = L_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_
        l_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_0_parameters_weight_ = L_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_0_parameters_weight_
        l_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_1_parameters_weight_ = L_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_1_parameters_weight_
        l_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_1_parameters_bias_ = L_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_1_parameters_bias_
        l_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_2_parameters_weight_ = L_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_2_parameters_weight_
        l_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_3_buffers_running_mean_ = L_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_3_buffers_running_mean_
        l_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_3_buffers_running_var_ = L_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_3_buffers_running_var_
        l_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_3_parameters_weight_ = L_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_3_parameters_weight_
        l_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_3_parameters_bias_ = L_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_3_parameters_bias_
        l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_ = L_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_
        l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_ = L_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_
        l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_ = L_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_
        l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_ = L_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_
        l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_ = L_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_
        l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_ = L_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_
        l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_ = L_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_
        l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_ = L_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_
        l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_ = L_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_
        l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_ = L_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_
        l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_ = L_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_
        l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_ = L_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_
        l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_ = L_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_
        l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_ = L_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_
        l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_ = L_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_
        l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_ = L_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_
        l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_ = L_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_
        l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_ = L_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_
        l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_ = L_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_
        l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_ = L_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_
        l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_ = L_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_
        l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_ = L_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_
        l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_ = L_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_
        l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_ = L_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_
        l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_ = L_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_
        l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_ = L_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_
        l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_ = L_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_
        l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_ = L_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_
        l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_ = L_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_
        l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_ = L_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_
        l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_ = L_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_
        l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_ = L_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_
        l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_ = L_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_
        l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_ = L_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_
        l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_ = L_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_
        l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_ = L_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_
        l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_ = L_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_
        l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_ = L_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_
        l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_ = L_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_
        l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_ = L_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_
        l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_ = L_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_
        l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_ = L_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_
        l_self_modules_blocks_modules_6_modules_3_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_6_modules_3_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_6_modules_3_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_6_modules_3_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_6_modules_3_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_6_modules_3_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_6_modules_3_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_6_modules_3_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_ = L_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_
        l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_ = L_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_
        l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_ = L_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_
        l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_ = L_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_
        l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_ = L_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_
        l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_ = L_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_
        l_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_0_parameters_weight_ = L_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_0_parameters_weight_
        l_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_1_parameters_weight_ = L_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_1_parameters_weight_
        l_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_1_parameters_bias_ = L_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_1_parameters_bias_
        l_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_2_parameters_weight_ = L_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_2_parameters_weight_
        l_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_3_buffers_running_mean_ = L_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_3_buffers_running_mean_
        l_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_3_buffers_running_var_ = L_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_3_buffers_running_var_
        l_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_3_parameters_weight_ = L_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_3_parameters_weight_
        l_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_3_parameters_bias_ = L_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_3_parameters_bias_
        l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_ = L_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_
        l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_ = L_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_
        l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_ = L_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_
        l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_ = L_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_
        l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_ = L_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_
        l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_ = L_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_
        l_self_modules_blocks_modules_6_modules_4_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_6_modules_4_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_6_modules_4_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_6_modules_4_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_6_modules_4_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_6_modules_4_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_6_modules_4_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_6_modules_4_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_ = L_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_
        l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_ = L_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_
        l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_ = L_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_
        l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_ = L_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_
        l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_ = L_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_
        l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_ = L_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_
        l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_ = L_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_
        l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_ = L_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_
        l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_ = L_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_
        l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_ = L_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_
        l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_ = L_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_
        l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_ = L_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_
        l_self_modules_blocks_modules_7_modules_0_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_7_modules_0_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_7_modules_0_modules_bn_dw_buffers_running_mean_ = L_self_modules_blocks_modules_7_modules_0_modules_bn_dw_buffers_running_mean_
        l_self_modules_blocks_modules_7_modules_0_modules_bn_dw_buffers_running_var_ = (
            L_self_modules_blocks_modules_7_modules_0_modules_bn_dw_buffers_running_var_
        )
        l_self_modules_blocks_modules_7_modules_0_modules_bn_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_7_modules_0_modules_bn_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_7_modules_0_modules_bn_dw_parameters_bias_ = (
            L_self_modules_blocks_modules_7_modules_0_modules_bn_dw_parameters_bias_
        )
        l_self_modules_blocks_modules_7_modules_0_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_7_modules_0_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_7_modules_0_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_7_modules_0_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_7_modules_0_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_7_modules_0_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_7_modules_0_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_7_modules_0_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_ = L_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_
        l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_ = L_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_
        l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_ = L_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_
        l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_ = L_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_
        l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_ = L_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_
        l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_ = L_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_
        l_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_0_parameters_weight_ = L_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_0_parameters_weight_
        l_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_1_parameters_weight_ = L_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_1_parameters_weight_
        l_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_1_parameters_bias_ = L_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_1_parameters_bias_
        l_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_2_parameters_weight_ = L_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_2_parameters_weight_
        l_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_3_buffers_running_mean_ = L_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_3_buffers_running_mean_
        l_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_3_buffers_running_var_ = L_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_3_buffers_running_var_
        l_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_3_parameters_weight_ = L_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_3_parameters_weight_
        l_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_3_parameters_bias_ = L_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_3_parameters_bias_
        l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_ = L_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_
        l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_ = L_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_
        l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_ = L_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_
        l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_ = L_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_
        l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_ = L_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_
        l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_ = L_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_
        l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_ = L_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_
        l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_ = L_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_
        l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_ = L_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_
        l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_ = L_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_
        l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_ = L_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_
        l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_ = L_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_
        l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_ = L_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_
        l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_ = L_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_
        l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_ = L_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_
        l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_ = L_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_
        l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_ = L_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_
        l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_ = L_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_
        l_self_modules_blocks_modules_8_modules_1_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_8_modules_1_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_8_modules_1_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_8_modules_1_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_8_modules_1_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_8_modules_1_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_8_modules_1_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_8_modules_1_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_ = L_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_
        l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_ = L_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_
        l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_ = L_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_
        l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_ = L_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_
        l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_ = L_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_
        l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_ = L_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_
        l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_ = L_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_
        l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_ = L_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_
        l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_ = L_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_
        l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_ = L_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_
        l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_ = L_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_
        l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_ = L_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_
        l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_ = L_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_
        l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_ = L_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_
        l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_ = L_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_
        l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_ = L_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_
        l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_ = L_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_
        l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_ = L_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_
        l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_ = L_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_
        l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_ = L_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_
        l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_ = L_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_
        l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_ = L_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_
        l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_ = L_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_
        l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_ = L_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_
        l_self_modules_blocks_modules_8_modules_3_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_8_modules_3_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_8_modules_3_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_8_modules_3_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_8_modules_3_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_8_modules_3_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_8_modules_3_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_8_modules_3_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_ = L_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_
        l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_ = L_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_
        l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_ = L_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_
        l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_ = L_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_
        l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_ = L_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_
        l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_ = L_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_
        l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_ = L_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_
        l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_ = L_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_
        l_self_modules_blocks_modules_9_modules_0_modules_conv_parameters_weight_ = (
            L_self_modules_blocks_modules_9_modules_0_modules_conv_parameters_weight_
        )
        l_self_modules_blocks_modules_9_modules_0_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_9_modules_0_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_9_modules_0_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_9_modules_0_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_9_modules_0_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_9_modules_0_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_9_modules_0_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_9_modules_0_modules_bn1_parameters_bias_
        )
        l_self_modules_conv_head_parameters_weight_ = (
            L_self_modules_conv_head_parameters_weight_
        )
        l_self_modules_conv_head_parameters_bias_ = (
            L_self_modules_conv_head_parameters_bias_
        )
        l_self_modules_classifier_parameters_weight_ = (
            L_self_modules_classifier_parameters_weight_
        )
        l_self_modules_classifier_parameters_bias_ = (
            L_self_modules_classifier_parameters_bias_
        )
        x = torch.conv2d(
            l_x_,
            l_self_modules_conv_stem_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_x_ = l_self_modules_conv_stem_parameters_weight_ = None
        x_1 = torch.nn.functional.batch_norm(
            x,
            l_self_modules_bn1_buffers_running_mean_,
            l_self_modules_bn1_buffers_running_var_,
            l_self_modules_bn1_parameters_weight_,
            l_self_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x = (
            l_self_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_bn1_parameters_weight_
        ) = l_self_modules_bn1_parameters_bias_ = None
        x_2 = torch.nn.functional.relu(x_1, inplace=True)
        x_1 = None
        input_1 = torch.conv2d(
            x_2,
            l_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_ = (
            None
        )
        input_2 = torch.nn.functional.batch_norm(
            input_1,
            l_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_1 = l_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_3 = torch.nn.functional.relu(input_2, inplace=True)
        input_2 = None
        input_4 = torch.conv2d(
            input_3,
            l_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        l_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_5 = torch.nn.functional.batch_norm(
            input_4,
            l_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_4 = l_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        input_6 = torch.nn.functional.relu(input_5, inplace=True)
        input_5 = None
        out = torch.cat([input_3, input_6], dim=1)
        input_3 = input_6 = None
        x_3 = out[
            (
                slice(None, None, None),
                slice(None, 16, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out = None
        input_7 = torch.conv2d(
            x_3,
            l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_3 = l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_ = (None)
        input_8 = torch.nn.functional.batch_norm(
            input_7,
            l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_7 = l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_9 = torch.conv2d(
            input_8,
            l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_10 = torch.nn.functional.batch_norm(
            input_9,
            l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_9 = l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        out_1 = torch.cat([input_8, input_10], dim=1)
        input_8 = input_10 = None
        x_4 = out_1[
            (
                slice(None, None, None),
                slice(None, 16, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out_1 = None
        x_4 += x_2
        x_5 = x_4
        x_4 = x_2 = None
        input_11 = torch.conv2d(
            x_5,
            l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_ = (
            None
        )
        input_12 = torch.nn.functional.batch_norm(
            input_11,
            l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_11 = l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_13 = torch.nn.functional.relu(input_12, inplace=True)
        input_12 = None
        input_14 = torch.conv2d(
            input_13,
            l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            24,
        )
        l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_15 = torch.nn.functional.batch_norm(
            input_14,
            l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_14 = l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        input_16 = torch.nn.functional.relu(input_15, inplace=True)
        input_15 = None
        out_2 = torch.cat([input_13, input_16], dim=1)
        input_13 = input_16 = None
        x_6 = out_2[
            (
                slice(None, None, None),
                slice(None, 48, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out_2 = None
        x_7 = torch.conv2d(
            x_6,
            l_self_modules_blocks_modules_1_modules_0_modules_conv_dw_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            48,
        )
        x_6 = (
            l_self_modules_blocks_modules_1_modules_0_modules_conv_dw_parameters_weight_
        ) = None
        x_8 = torch.nn.functional.batch_norm(
            x_7,
            l_self_modules_blocks_modules_1_modules_0_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_7 = l_self_modules_blocks_modules_1_modules_0_modules_bn_dw_buffers_running_mean_ = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn_dw_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn_dw_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn_dw_parameters_bias_
        ) = None
        input_17 = torch.conv2d(
            x_8,
            l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_8 = l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_ = (None)
        input_18 = torch.nn.functional.batch_norm(
            input_17,
            l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_17 = l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_19 = torch.conv2d(
            input_18,
            l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            12,
        )
        l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_20 = torch.nn.functional.batch_norm(
            input_19,
            l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_19 = l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        out_3 = torch.cat([input_18, input_20], dim=1)
        input_18 = input_20 = None
        x_9 = out_3[
            (
                slice(None, None, None),
                slice(None, 24, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out_3 = None
        input_21 = torch.conv2d(
            x_5,
            l_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            16,
        )
        x_5 = l_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_0_parameters_weight_ = (None)
        input_22 = torch.nn.functional.batch_norm(
            input_21,
            l_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_21 = l_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_1_parameters_weight_ = l_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_1_parameters_bias_ = (None)
        input_23 = torch.conv2d(
            input_22,
            l_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_22 = l_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_2_parameters_weight_ = (None)
        input_24 = torch.nn.functional.batch_norm(
            input_23,
            l_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_3_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_3_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_3_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_23 = l_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_3_buffers_running_mean_ = l_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_3_buffers_running_var_ = l_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_3_parameters_weight_ = l_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_3_parameters_bias_ = (None)
        x_9 += input_24
        x_10 = x_9
        x_9 = input_24 = None
        input_25 = torch.conv2d(
            x_10,
            l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_ = (
            None
        )
        input_26 = torch.nn.functional.batch_norm(
            input_25,
            l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_25 = l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_27 = torch.nn.functional.relu(input_26, inplace=True)
        input_26 = None
        input_28 = torch.conv2d(
            input_27,
            l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            36,
        )
        l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_29 = torch.nn.functional.batch_norm(
            input_28,
            l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_28 = l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        input_30 = torch.nn.functional.relu(input_29, inplace=True)
        input_29 = None
        out_4 = torch.cat([input_27, input_30], dim=1)
        input_27 = input_30 = None
        x_11 = out_4[
            (
                slice(None, None, None),
                slice(None, 72, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out_4 = None
        input_31 = torch.conv2d(
            x_11,
            l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_11 = l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_ = (None)
        input_32 = torch.nn.functional.batch_norm(
            input_31,
            l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_31 = l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_33 = torch.conv2d(
            input_32,
            l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            12,
        )
        l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_34 = torch.nn.functional.batch_norm(
            input_33,
            l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_33 = l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        out_5 = torch.cat([input_32, input_34], dim=1)
        input_32 = input_34 = None
        x_12 = out_5[
            (
                slice(None, None, None),
                slice(None, 24, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out_5 = None
        x_12 += x_10
        x_13 = x_12
        x_12 = x_10 = None
        input_35 = torch.conv2d(
            x_13,
            l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_ = (
            None
        )
        input_36 = torch.nn.functional.batch_norm(
            input_35,
            l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_35 = l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_37 = torch.nn.functional.relu(input_36, inplace=True)
        input_36 = None
        input_38 = torch.conv2d(
            input_37,
            l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            36,
        )
        l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_39 = torch.nn.functional.batch_norm(
            input_38,
            l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_38 = l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        input_40 = torch.nn.functional.relu(input_39, inplace=True)
        input_39 = None
        out_6 = torch.cat([input_37, input_40], dim=1)
        input_37 = input_40 = None
        x_14 = out_6[
            (
                slice(None, None, None),
                slice(None, 72, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out_6 = None
        x_15 = torch.conv2d(
            x_14,
            l_self_modules_blocks_modules_3_modules_0_modules_conv_dw_parameters_weight_,
            None,
            (2, 2),
            (2, 2),
            (1, 1),
            72,
        )
        x_14 = (
            l_self_modules_blocks_modules_3_modules_0_modules_conv_dw_parameters_weight_
        ) = None
        x_16 = torch.nn.functional.batch_norm(
            x_15,
            l_self_modules_blocks_modules_3_modules_0_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_15 = l_self_modules_blocks_modules_3_modules_0_modules_bn_dw_buffers_running_mean_ = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn_dw_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn_dw_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn_dw_parameters_bias_
        ) = None
        x_se = x_16.mean((2, 3), keepdim=True)
        x_se_1 = torch.conv2d(
            x_se,
            l_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se = l_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_2 = torch.nn.functional.relu(x_se_1, inplace=True)
        x_se_1 = None
        x_se_3 = torch.conv2d(
            x_se_2,
            l_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_2 = l_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_expand_parameters_bias_ = (None)
        hardsigmoid = torch.nn.functional.hardsigmoid(x_se_3, False)
        x_se_3 = None
        x_17 = x_16 * hardsigmoid
        x_16 = hardsigmoid = None
        input_41 = torch.conv2d(
            x_17,
            l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_17 = l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_ = (None)
        input_42 = torch.nn.functional.batch_norm(
            input_41,
            l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_41 = l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_43 = torch.conv2d(
            input_42,
            l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            20,
        )
        l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_44 = torch.nn.functional.batch_norm(
            input_43,
            l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_43 = l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        out_7 = torch.cat([input_42, input_44], dim=1)
        input_42 = input_44 = None
        x_18 = out_7[
            (
                slice(None, None, None),
                slice(None, 40, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out_7 = None
        input_45 = torch.conv2d(
            x_13,
            l_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_0_parameters_weight_,
            None,
            (2, 2),
            (2, 2),
            (1, 1),
            24,
        )
        x_13 = l_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_0_parameters_weight_ = (None)
        input_46 = torch.nn.functional.batch_norm(
            input_45,
            l_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_45 = l_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_1_parameters_weight_ = l_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_1_parameters_bias_ = (None)
        input_47 = torch.conv2d(
            input_46,
            l_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_46 = l_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_2_parameters_weight_ = (None)
        input_48 = torch.nn.functional.batch_norm(
            input_47,
            l_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_3_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_3_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_3_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_47 = l_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_3_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_3_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_3_parameters_weight_ = l_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_3_parameters_bias_ = (None)
        x_18 += input_48
        x_19 = x_18
        x_18 = input_48 = None
        input_49 = torch.conv2d(
            x_19,
            l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_ = (
            None
        )
        input_50 = torch.nn.functional.batch_norm(
            input_49,
            l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_49 = l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_51 = torch.nn.functional.relu(input_50, inplace=True)
        input_50 = None
        input_52 = torch.conv2d(
            input_51,
            l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            60,
        )
        l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_53 = torch.nn.functional.batch_norm(
            input_52,
            l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_52 = l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        input_54 = torch.nn.functional.relu(input_53, inplace=True)
        input_53 = None
        out_8 = torch.cat([input_51, input_54], dim=1)
        input_51 = input_54 = None
        x_20 = out_8[
            (
                slice(None, None, None),
                slice(None, 120, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out_8 = None
        x_se_4 = x_20.mean((2, 3), keepdim=True)
        x_se_5 = torch.conv2d(
            x_se_4,
            l_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_4 = l_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_6 = torch.nn.functional.relu(x_se_5, inplace=True)
        x_se_5 = None
        x_se_7 = torch.conv2d(
            x_se_6,
            l_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_6 = l_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_expand_parameters_bias_ = (None)
        hardsigmoid_1 = torch.nn.functional.hardsigmoid(x_se_7, False)
        x_se_7 = None
        x_21 = x_20 * hardsigmoid_1
        x_20 = hardsigmoid_1 = None
        input_55 = torch.conv2d(
            x_21,
            l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_21 = l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_ = (None)
        input_56 = torch.nn.functional.batch_norm(
            input_55,
            l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_55 = l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_57 = torch.conv2d(
            input_56,
            l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            20,
        )
        l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_58 = torch.nn.functional.batch_norm(
            input_57,
            l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_57 = l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        out_9 = torch.cat([input_56, input_58], dim=1)
        input_56 = input_58 = None
        x_22 = out_9[
            (
                slice(None, None, None),
                slice(None, 40, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out_9 = None
        x_22 += x_19
        x_23 = x_22
        x_22 = x_19 = None
        input_59 = torch.conv2d(
            x_23,
            l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_ = (
            None
        )
        input_60 = torch.nn.functional.batch_norm(
            input_59,
            l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_59 = l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_61 = torch.nn.functional.relu(input_60, inplace=True)
        input_60 = None
        input_62 = torch.conv2d(
            input_61,
            l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            120,
        )
        l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_63 = torch.nn.functional.batch_norm(
            input_62,
            l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_62 = l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        input_64 = torch.nn.functional.relu(input_63, inplace=True)
        input_63 = None
        out_10 = torch.cat([input_61, input_64], dim=1)
        input_61 = input_64 = None
        x_24 = out_10[
            (
                slice(None, None, None),
                slice(None, 240, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out_10 = None
        x_25 = torch.conv2d(
            x_24,
            l_self_modules_blocks_modules_5_modules_0_modules_conv_dw_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            240,
        )
        x_24 = (
            l_self_modules_blocks_modules_5_modules_0_modules_conv_dw_parameters_weight_
        ) = None
        x_26 = torch.nn.functional.batch_norm(
            x_25,
            l_self_modules_blocks_modules_5_modules_0_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_25 = l_self_modules_blocks_modules_5_modules_0_modules_bn_dw_buffers_running_mean_ = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn_dw_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn_dw_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn_dw_parameters_bias_
        ) = None
        input_65 = torch.conv2d(
            x_26,
            l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_26 = l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_ = (None)
        input_66 = torch.nn.functional.batch_norm(
            input_65,
            l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_65 = l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_67 = torch.conv2d(
            input_66,
            l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            40,
        )
        l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_68 = torch.nn.functional.batch_norm(
            input_67,
            l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_67 = l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        out_11 = torch.cat([input_66, input_68], dim=1)
        input_66 = input_68 = None
        x_27 = out_11[
            (
                slice(None, None, None),
                slice(None, 80, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out_11 = None
        input_69 = torch.conv2d(
            x_23,
            l_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            40,
        )
        x_23 = l_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_0_parameters_weight_ = (None)
        input_70 = torch.nn.functional.batch_norm(
            input_69,
            l_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_69 = l_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_1_parameters_weight_ = l_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_1_parameters_bias_ = (None)
        input_71 = torch.conv2d(
            input_70,
            l_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_70 = l_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_2_parameters_weight_ = (None)
        input_72 = torch.nn.functional.batch_norm(
            input_71,
            l_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_71 = l_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_3_buffers_running_mean_ = l_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_3_buffers_running_var_ = l_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_3_parameters_weight_ = l_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_3_parameters_bias_ = (None)
        x_27 += input_72
        x_28 = x_27
        x_27 = input_72 = None
        input_73 = torch.conv2d(
            x_28,
            l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_ = (
            None
        )
        input_74 = torch.nn.functional.batch_norm(
            input_73,
            l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_73 = l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_75 = torch.nn.functional.relu(input_74, inplace=True)
        input_74 = None
        input_76 = torch.conv2d(
            input_75,
            l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            100,
        )
        l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_77 = torch.nn.functional.batch_norm(
            input_76,
            l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_76 = l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        input_78 = torch.nn.functional.relu(input_77, inplace=True)
        input_77 = None
        out_12 = torch.cat([input_75, input_78], dim=1)
        input_75 = input_78 = None
        x_29 = out_12[
            (
                slice(None, None, None),
                slice(None, 200, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out_12 = None
        input_79 = torch.conv2d(
            x_29,
            l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_29 = l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_ = (None)
        input_80 = torch.nn.functional.batch_norm(
            input_79,
            l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_79 = l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_81 = torch.conv2d(
            input_80,
            l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            40,
        )
        l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_82 = torch.nn.functional.batch_norm(
            input_81,
            l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_81 = l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        out_13 = torch.cat([input_80, input_82], dim=1)
        input_80 = input_82 = None
        x_30 = out_13[
            (
                slice(None, None, None),
                slice(None, 80, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out_13 = None
        x_30 += x_28
        x_31 = x_30
        x_30 = x_28 = None
        input_83 = torch.conv2d(
            x_31,
            l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_ = (
            None
        )
        input_84 = torch.nn.functional.batch_norm(
            input_83,
            l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_83 = l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_85 = torch.nn.functional.relu(input_84, inplace=True)
        input_84 = None
        input_86 = torch.conv2d(
            input_85,
            l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            92,
        )
        l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_87 = torch.nn.functional.batch_norm(
            input_86,
            l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_86 = l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        input_88 = torch.nn.functional.relu(input_87, inplace=True)
        input_87 = None
        out_14 = torch.cat([input_85, input_88], dim=1)
        input_85 = input_88 = None
        x_32 = out_14[
            (
                slice(None, None, None),
                slice(None, 184, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out_14 = None
        input_89 = torch.conv2d(
            x_32,
            l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_32 = l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_ = (None)
        input_90 = torch.nn.functional.batch_norm(
            input_89,
            l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_89 = l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_91 = torch.conv2d(
            input_90,
            l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            40,
        )
        l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_92 = torch.nn.functional.batch_norm(
            input_91,
            l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_91 = l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        out_15 = torch.cat([input_90, input_92], dim=1)
        input_90 = input_92 = None
        x_33 = out_15[
            (
                slice(None, None, None),
                slice(None, 80, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out_15 = None
        x_33 += x_31
        x_34 = x_33
        x_33 = x_31 = None
        input_93 = torch.conv2d(
            x_34,
            l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_ = (
            None
        )
        input_94 = torch.nn.functional.batch_norm(
            input_93,
            l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_93 = l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_95 = torch.nn.functional.relu(input_94, inplace=True)
        input_94 = None
        input_96 = torch.conv2d(
            input_95,
            l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            92,
        )
        l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_97 = torch.nn.functional.batch_norm(
            input_96,
            l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_96 = l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        input_98 = torch.nn.functional.relu(input_97, inplace=True)
        input_97 = None
        out_16 = torch.cat([input_95, input_98], dim=1)
        input_95 = input_98 = None
        x_35 = out_16[
            (
                slice(None, None, None),
                slice(None, 184, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out_16 = None
        input_99 = torch.conv2d(
            x_35,
            l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_35 = l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_ = (None)
        input_100 = torch.nn.functional.batch_norm(
            input_99,
            l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_99 = l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_101 = torch.conv2d(
            input_100,
            l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            40,
        )
        l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_102 = torch.nn.functional.batch_norm(
            input_101,
            l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_101 = l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        out_17 = torch.cat([input_100, input_102], dim=1)
        input_100 = input_102 = None
        x_36 = out_17[
            (
                slice(None, None, None),
                slice(None, 80, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out_17 = None
        x_36 += x_34
        x_37 = x_36
        x_36 = x_34 = None
        input_103 = torch.conv2d(
            x_37,
            l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_ = (
            None
        )
        input_104 = torch.nn.functional.batch_norm(
            input_103,
            l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_103 = l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_105 = torch.nn.functional.relu(input_104, inplace=True)
        input_104 = None
        input_106 = torch.conv2d(
            input_105,
            l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            240,
        )
        l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_107 = torch.nn.functional.batch_norm(
            input_106,
            l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_106 = l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        input_108 = torch.nn.functional.relu(input_107, inplace=True)
        input_107 = None
        out_18 = torch.cat([input_105, input_108], dim=1)
        input_105 = input_108 = None
        x_38 = out_18[
            (
                slice(None, None, None),
                slice(None, 480, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out_18 = None
        x_se_8 = x_38.mean((2, 3), keepdim=True)
        x_se_9 = torch.conv2d(
            x_se_8,
            l_self_modules_blocks_modules_6_modules_3_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_3_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_8 = l_self_modules_blocks_modules_6_modules_3_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_6_modules_3_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_10 = torch.nn.functional.relu(x_se_9, inplace=True)
        x_se_9 = None
        x_se_11 = torch.conv2d(
            x_se_10,
            l_self_modules_blocks_modules_6_modules_3_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_3_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_10 = l_self_modules_blocks_modules_6_modules_3_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_6_modules_3_modules_se_modules_conv_expand_parameters_bias_ = (None)
        hardsigmoid_2 = torch.nn.functional.hardsigmoid(x_se_11, False)
        x_se_11 = None
        x_39 = x_38 * hardsigmoid_2
        x_38 = hardsigmoid_2 = None
        input_109 = torch.conv2d(
            x_39,
            l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_39 = l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_ = (None)
        input_110 = torch.nn.functional.batch_norm(
            input_109,
            l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_109 = l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_111 = torch.conv2d(
            input_110,
            l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            56,
        )
        l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_112 = torch.nn.functional.batch_norm(
            input_111,
            l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_111 = l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        out_19 = torch.cat([input_110, input_112], dim=1)
        input_110 = input_112 = None
        x_40 = out_19[
            (
                slice(None, None, None),
                slice(None, 112, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out_19 = None
        input_113 = torch.conv2d(
            x_37,
            l_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            80,
        )
        x_37 = l_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_0_parameters_weight_ = (None)
        input_114 = torch.nn.functional.batch_norm(
            input_113,
            l_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_113 = l_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_1_parameters_weight_ = l_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_1_parameters_bias_ = (None)
        input_115 = torch.conv2d(
            input_114,
            l_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_114 = l_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_2_parameters_weight_ = (None)
        input_116 = torch.nn.functional.batch_norm(
            input_115,
            l_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_3_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_3_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_3_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_115 = l_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_3_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_3_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_3_parameters_weight_ = l_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_3_parameters_bias_ = (None)
        x_40 += input_116
        x_41 = x_40
        x_40 = input_116 = None
        input_117 = torch.conv2d(
            x_41,
            l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_ = (
            None
        )
        input_118 = torch.nn.functional.batch_norm(
            input_117,
            l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_117 = l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_119 = torch.nn.functional.relu(input_118, inplace=True)
        input_118 = None
        input_120 = torch.conv2d(
            input_119,
            l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            336,
        )
        l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_121 = torch.nn.functional.batch_norm(
            input_120,
            l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_120 = l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        input_122 = torch.nn.functional.relu(input_121, inplace=True)
        input_121 = None
        out_20 = torch.cat([input_119, input_122], dim=1)
        input_119 = input_122 = None
        x_42 = out_20[
            (
                slice(None, None, None),
                slice(None, 672, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out_20 = None
        x_se_12 = x_42.mean((2, 3), keepdim=True)
        x_se_13 = torch.conv2d(
            x_se_12,
            l_self_modules_blocks_modules_6_modules_4_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_4_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_12 = l_self_modules_blocks_modules_6_modules_4_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_6_modules_4_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_14 = torch.nn.functional.relu(x_se_13, inplace=True)
        x_se_13 = None
        x_se_15 = torch.conv2d(
            x_se_14,
            l_self_modules_blocks_modules_6_modules_4_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_4_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_14 = l_self_modules_blocks_modules_6_modules_4_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_6_modules_4_modules_se_modules_conv_expand_parameters_bias_ = (None)
        hardsigmoid_3 = torch.nn.functional.hardsigmoid(x_se_15, False)
        x_se_15 = None
        x_43 = x_42 * hardsigmoid_3
        x_42 = hardsigmoid_3 = None
        input_123 = torch.conv2d(
            x_43,
            l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_43 = l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_ = (None)
        input_124 = torch.nn.functional.batch_norm(
            input_123,
            l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_123 = l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_125 = torch.conv2d(
            input_124,
            l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            56,
        )
        l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_126 = torch.nn.functional.batch_norm(
            input_125,
            l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_125 = l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        out_21 = torch.cat([input_124, input_126], dim=1)
        input_124 = input_126 = None
        x_44 = out_21[
            (
                slice(None, None, None),
                slice(None, 112, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out_21 = None
        x_44 += x_41
        x_45 = x_44
        x_44 = x_41 = None
        input_127 = torch.conv2d(
            x_45,
            l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_ = (
            None
        )
        input_128 = torch.nn.functional.batch_norm(
            input_127,
            l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_127 = l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_129 = torch.nn.functional.relu(input_128, inplace=True)
        input_128 = None
        input_130 = torch.conv2d(
            input_129,
            l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            336,
        )
        l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_131 = torch.nn.functional.batch_norm(
            input_130,
            l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_130 = l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        input_132 = torch.nn.functional.relu(input_131, inplace=True)
        input_131 = None
        out_22 = torch.cat([input_129, input_132], dim=1)
        input_129 = input_132 = None
        x_46 = out_22[
            (
                slice(None, None, None),
                slice(None, 672, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out_22 = None
        x_47 = torch.conv2d(
            x_46,
            l_self_modules_blocks_modules_7_modules_0_modules_conv_dw_parameters_weight_,
            None,
            (2, 2),
            (2, 2),
            (1, 1),
            672,
        )
        x_46 = (
            l_self_modules_blocks_modules_7_modules_0_modules_conv_dw_parameters_weight_
        ) = None
        x_48 = torch.nn.functional.batch_norm(
            x_47,
            l_self_modules_blocks_modules_7_modules_0_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_7_modules_0_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_7_modules_0_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_0_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_47 = l_self_modules_blocks_modules_7_modules_0_modules_bn_dw_buffers_running_mean_ = (
            l_self_modules_blocks_modules_7_modules_0_modules_bn_dw_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_7_modules_0_modules_bn_dw_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_7_modules_0_modules_bn_dw_parameters_bias_
        ) = None
        x_se_16 = x_48.mean((2, 3), keepdim=True)
        x_se_17 = torch.conv2d(
            x_se_16,
            l_self_modules_blocks_modules_7_modules_0_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_0_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_16 = l_self_modules_blocks_modules_7_modules_0_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_7_modules_0_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_18 = torch.nn.functional.relu(x_se_17, inplace=True)
        x_se_17 = None
        x_se_19 = torch.conv2d(
            x_se_18,
            l_self_modules_blocks_modules_7_modules_0_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_0_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_18 = l_self_modules_blocks_modules_7_modules_0_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_7_modules_0_modules_se_modules_conv_expand_parameters_bias_ = (None)
        hardsigmoid_4 = torch.nn.functional.hardsigmoid(x_se_19, False)
        x_se_19 = None
        x_49 = x_48 * hardsigmoid_4
        x_48 = hardsigmoid_4 = None
        input_133 = torch.conv2d(
            x_49,
            l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_49 = l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_ = (None)
        input_134 = torch.nn.functional.batch_norm(
            input_133,
            l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_133 = l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_135 = torch.conv2d(
            input_134,
            l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            80,
        )
        l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_136 = torch.nn.functional.batch_norm(
            input_135,
            l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_135 = l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        out_23 = torch.cat([input_134, input_136], dim=1)
        input_134 = input_136 = None
        x_50 = out_23[
            (
                slice(None, None, None),
                slice(None, 160, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out_23 = None
        input_137 = torch.conv2d(
            x_45,
            l_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_0_parameters_weight_,
            None,
            (2, 2),
            (2, 2),
            (1, 1),
            112,
        )
        x_45 = l_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_0_parameters_weight_ = (None)
        input_138 = torch.nn.functional.batch_norm(
            input_137,
            l_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_137 = l_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_1_parameters_weight_ = l_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_1_parameters_bias_ = (None)
        input_139 = torch.conv2d(
            input_138,
            l_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_138 = l_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_2_parameters_weight_ = (None)
        input_140 = torch.nn.functional.batch_norm(
            input_139,
            l_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_3_buffers_running_mean_,
            l_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_3_buffers_running_var_,
            l_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_3_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_139 = l_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_3_buffers_running_mean_ = l_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_3_buffers_running_var_ = l_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_3_parameters_weight_ = l_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_3_parameters_bias_ = (None)
        x_50 += input_140
        x_51 = x_50
        x_50 = input_140 = None
        input_141 = torch.conv2d(
            x_51,
            l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_ = (
            None
        )
        input_142 = torch.nn.functional.batch_norm(
            input_141,
            l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_141 = l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_143 = torch.nn.functional.relu(input_142, inplace=True)
        input_142 = None
        input_144 = torch.conv2d(
            input_143,
            l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            480,
        )
        l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_145 = torch.nn.functional.batch_norm(
            input_144,
            l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_144 = l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        input_146 = torch.nn.functional.relu(input_145, inplace=True)
        input_145 = None
        out_24 = torch.cat([input_143, input_146], dim=1)
        input_143 = input_146 = None
        x_52 = out_24[
            (
                slice(None, None, None),
                slice(None, 960, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out_24 = None
        input_147 = torch.conv2d(
            x_52,
            l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_52 = l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_ = (None)
        input_148 = torch.nn.functional.batch_norm(
            input_147,
            l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_147 = l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_149 = torch.conv2d(
            input_148,
            l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            80,
        )
        l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_150 = torch.nn.functional.batch_norm(
            input_149,
            l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_149 = l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        out_25 = torch.cat([input_148, input_150], dim=1)
        input_148 = input_150 = None
        x_53 = out_25[
            (
                slice(None, None, None),
                slice(None, 160, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out_25 = None
        x_53 += x_51
        x_54 = x_53
        x_53 = x_51 = None
        input_151 = torch.conv2d(
            x_54,
            l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_ = (
            None
        )
        input_152 = torch.nn.functional.batch_norm(
            input_151,
            l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_151 = l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_153 = torch.nn.functional.relu(input_152, inplace=True)
        input_152 = None
        input_154 = torch.conv2d(
            input_153,
            l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            480,
        )
        l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_155 = torch.nn.functional.batch_norm(
            input_154,
            l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_154 = l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        input_156 = torch.nn.functional.relu(input_155, inplace=True)
        input_155 = None
        out_26 = torch.cat([input_153, input_156], dim=1)
        input_153 = input_156 = None
        x_55 = out_26[
            (
                slice(None, None, None),
                slice(None, 960, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out_26 = None
        x_se_20 = x_55.mean((2, 3), keepdim=True)
        x_se_21 = torch.conv2d(
            x_se_20,
            l_self_modules_blocks_modules_8_modules_1_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_1_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_20 = l_self_modules_blocks_modules_8_modules_1_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_8_modules_1_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_22 = torch.nn.functional.relu(x_se_21, inplace=True)
        x_se_21 = None
        x_se_23 = torch.conv2d(
            x_se_22,
            l_self_modules_blocks_modules_8_modules_1_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_1_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_22 = l_self_modules_blocks_modules_8_modules_1_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_8_modules_1_modules_se_modules_conv_expand_parameters_bias_ = (None)
        hardsigmoid_5 = torch.nn.functional.hardsigmoid(x_se_23, False)
        x_se_23 = None
        x_56 = x_55 * hardsigmoid_5
        x_55 = hardsigmoid_5 = None
        input_157 = torch.conv2d(
            x_56,
            l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_56 = l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_ = (None)
        input_158 = torch.nn.functional.batch_norm(
            input_157,
            l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_157 = l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_159 = torch.conv2d(
            input_158,
            l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            80,
        )
        l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_160 = torch.nn.functional.batch_norm(
            input_159,
            l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_159 = l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        out_27 = torch.cat([input_158, input_160], dim=1)
        input_158 = input_160 = None
        x_57 = out_27[
            (
                slice(None, None, None),
                slice(None, 160, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out_27 = None
        x_57 += x_54
        x_58 = x_57
        x_57 = x_54 = None
        input_161 = torch.conv2d(
            x_58,
            l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_ = (
            None
        )
        input_162 = torch.nn.functional.batch_norm(
            input_161,
            l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_161 = l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_163 = torch.nn.functional.relu(input_162, inplace=True)
        input_162 = None
        input_164 = torch.conv2d(
            input_163,
            l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            480,
        )
        l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_165 = torch.nn.functional.batch_norm(
            input_164,
            l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_164 = l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        input_166 = torch.nn.functional.relu(input_165, inplace=True)
        input_165 = None
        out_28 = torch.cat([input_163, input_166], dim=1)
        input_163 = input_166 = None
        x_59 = out_28[
            (
                slice(None, None, None),
                slice(None, 960, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out_28 = None
        input_167 = torch.conv2d(
            x_59,
            l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_59 = l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_ = (None)
        input_168 = torch.nn.functional.batch_norm(
            input_167,
            l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_167 = l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_169 = torch.conv2d(
            input_168,
            l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            80,
        )
        l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_170 = torch.nn.functional.batch_norm(
            input_169,
            l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_169 = l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        out_29 = torch.cat([input_168, input_170], dim=1)
        input_168 = input_170 = None
        x_60 = out_29[
            (
                slice(None, None, None),
                slice(None, 160, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out_29 = None
        x_60 += x_58
        x_61 = x_60
        x_60 = x_58 = None
        input_171 = torch.conv2d(
            x_61,
            l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_primary_conv_modules_0_parameters_weight_ = (
            None
        )
        input_172 = torch.nn.functional.batch_norm(
            input_171,
            l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_171 = l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_173 = torch.nn.functional.relu(input_172, inplace=True)
        input_172 = None
        input_174 = torch.conv2d(
            input_173,
            l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            480,
        )
        l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_175 = torch.nn.functional.batch_norm(
            input_174,
            l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_174 = l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        input_176 = torch.nn.functional.relu(input_175, inplace=True)
        input_175 = None
        out_30 = torch.cat([input_173, input_176], dim=1)
        input_173 = input_176 = None
        x_62 = out_30[
            (
                slice(None, None, None),
                slice(None, 960, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out_30 = None
        x_se_24 = x_62.mean((2, 3), keepdim=True)
        x_se_25 = torch.conv2d(
            x_se_24,
            l_self_modules_blocks_modules_8_modules_3_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_3_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_24 = l_self_modules_blocks_modules_8_modules_3_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_8_modules_3_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_26 = torch.nn.functional.relu(x_se_25, inplace=True)
        x_se_25 = None
        x_se_27 = torch.conv2d(
            x_se_26,
            l_self_modules_blocks_modules_8_modules_3_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_3_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_26 = l_self_modules_blocks_modules_8_modules_3_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_8_modules_3_modules_se_modules_conv_expand_parameters_bias_ = (None)
        hardsigmoid_6 = torch.nn.functional.hardsigmoid(x_se_27, False)
        x_se_27 = None
        x_63 = x_62 * hardsigmoid_6
        x_62 = hardsigmoid_6 = None
        input_177 = torch.conv2d(
            x_63,
            l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_63 = l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_ = (None)
        input_178 = torch.nn.functional.batch_norm(
            input_177,
            l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_177 = l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_179 = torch.conv2d(
            input_178,
            l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            80,
        )
        l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_180 = torch.nn.functional.batch_norm(
            input_179,
            l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_179 = l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        out_31 = torch.cat([input_178, input_180], dim=1)
        input_178 = input_180 = None
        x_64 = out_31[
            (
                slice(None, None, None),
                slice(None, 160, None),
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        out_31 = None
        x_64 += x_61
        x_65 = x_64
        x_64 = x_61 = None
        x_66 = torch.conv2d(
            x_65,
            l_self_modules_blocks_modules_9_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_65 = (
            l_self_modules_blocks_modules_9_modules_0_modules_conv_parameters_weight_
        ) = None
        x_67 = torch.nn.functional.batch_norm(
            x_66,
            l_self_modules_blocks_modules_9_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_9_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_9_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_66 = (
            l_self_modules_blocks_modules_9_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_9_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_9_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_9_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_68 = torch.nn.functional.relu(x_67, inplace=True)
        x_67 = None
        x_69 = torch.nn.functional.adaptive_avg_pool2d(x_68, 1)
        x_68 = None
        x_70 = torch.conv2d(
            x_69,
            l_self_modules_conv_head_parameters_weight_,
            l_self_modules_conv_head_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_69 = (
            l_self_modules_conv_head_parameters_weight_
        ) = l_self_modules_conv_head_parameters_bias_ = None
        x_71 = torch.nn.functional.relu(x_70, inplace=True)
        x_70 = None
        x_72 = x_71.flatten(1, -1)
        x_71 = None
        x_73 = torch.nn.functional.dropout(x_72, p=0.2, training=False)
        x_72 = None
        x_74 = torch._C._nn.linear(
            x_73,
            l_self_modules_classifier_parameters_weight_,
            l_self_modules_classifier_parameters_bias_,
        )
        x_73 = (
            l_self_modules_classifier_parameters_weight_
        ) = l_self_modules_classifier_parameters_bias_ = None
        return (x_74,)
