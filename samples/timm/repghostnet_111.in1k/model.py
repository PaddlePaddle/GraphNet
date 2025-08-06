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
        L_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_ = L_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_
        l_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_ = L_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_
        l_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_
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
        l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_ = L_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_
        l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_ = L_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_
        l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_
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
        l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_ = L_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_
        l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_ = L_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_
        l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_
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
        l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_ = L_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_
        l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_ = L_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_
        l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_
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
        l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_
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
        l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_
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
        l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_
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
        l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_
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
        l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_ = L_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_
        l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_ = L_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_
        l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_
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
        l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_ = L_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_
        l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_ = L_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_
        l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_
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
        l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_ = L_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_
        l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_ = L_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_
        l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_
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
        l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_ = L_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_
        l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_ = L_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_
        l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_
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
        l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_ = L_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_
        l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_ = L_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_
        l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_
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
        l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_ = L_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_
        l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_ = L_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_
        l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_
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
        l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_ = L_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_
        l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_ = L_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_
        l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_
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
        l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_ = L_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_
        l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_ = L_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_
        l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_
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
        l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_ = L_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_
        l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_ = L_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_
        l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_
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
        l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_ = L_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_
        l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_ = L_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_
        l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_
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
        l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_ = L_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_
        l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_ = L_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_
        l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_
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
        l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_ = L_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_
        l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_ = L_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_
        l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_
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
        l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_ = L_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_
        l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_ = L_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_
        l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_
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
        l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_ = L_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_
        l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_ = L_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_
        l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_
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
        l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_ = L_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_
        l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_ = L_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_
        l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_
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
        l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_ = L_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_
        l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_ = L_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_
        l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_
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
        l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_ = L_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_
        l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_ = L_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_
        l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_
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
        l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_ = L_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_
        l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_ = L_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_
        l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_
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
        l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_ = L_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_
        l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_ = L_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_
        l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_
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
        l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_ = L_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_
        l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_ = L_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_
        l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_
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
        l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_ = L_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_
        l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_ = L_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_
        l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_
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
        l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_ = L_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_
        l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_ = L_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_
        l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_
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
        l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_ = L_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_
        l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_ = L_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_
        l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_
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
        l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_ = L_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_
        l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_ = L_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_
        l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_
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
        batch_norm_3 = torch.nn.functional.batch_norm(
            input_3,
            l_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_,
            l_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_,
            l_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_3 = l_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_ = l_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_ = l_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_0_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_ = (None)
        x2 = input_5 + batch_norm_3
        input_5 = batch_norm_3 = None
        x_3 = torch.nn.functional.relu(x2, inplace=False)
        x2 = None
        input_6 = torch.conv2d(
            x_3,
            l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_3 = l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_ = (None)
        input_7 = torch.nn.functional.batch_norm(
            input_6,
            l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_6 = l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_8 = torch.conv2d(
            input_7,
            l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            16,
        )
        l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_9 = torch.nn.functional.batch_norm(
            input_8,
            l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_8 = l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        batch_norm_6 = torch.nn.functional.batch_norm(
            input_7,
            l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_,
            l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_,
            l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_7 = l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_ = l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_ = l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_0_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_ = (None)
        x2_1 = input_9 + batch_norm_6
        input_9 = batch_norm_6 = None
        x2_1 += x_2
        x_4 = x2_1
        x2_1 = x_2 = None
        input_10 = torch.conv2d(
            x_4,
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
        input_11 = torch.nn.functional.batch_norm(
            input_10,
            l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_10 = l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_12 = torch.nn.functional.relu(input_11, inplace=True)
        input_11 = None
        input_13 = torch.conv2d(
            input_12,
            l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            28,
        )
        l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_14 = torch.nn.functional.batch_norm(
            input_13,
            l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_13 = l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        batch_norm_9 = torch.nn.functional.batch_norm(
            input_12,
            l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_12 = l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_ = l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_ = l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_1_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_ = (None)
        x2_2 = input_14 + batch_norm_9
        input_14 = batch_norm_9 = None
        x_5 = torch.nn.functional.relu(x2_2, inplace=False)
        x2_2 = None
        x_6 = torch.conv2d(
            x_5,
            l_self_modules_blocks_modules_1_modules_0_modules_conv_dw_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            28,
        )
        x_5 = (
            l_self_modules_blocks_modules_1_modules_0_modules_conv_dw_parameters_weight_
        ) = None
        x_7 = torch.nn.functional.batch_norm(
            x_6,
            l_self_modules_blocks_modules_1_modules_0_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_6 = l_self_modules_blocks_modules_1_modules_0_modules_bn_dw_buffers_running_mean_ = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn_dw_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn_dw_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn_dw_parameters_bias_
        ) = None
        input_15 = torch.conv2d(
            x_7,
            l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_7 = l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_ = (None)
        input_16 = torch.nn.functional.batch_norm(
            input_15,
            l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_15 = l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_17 = torch.conv2d(
            input_16,
            l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            28,
        )
        l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_18 = torch.nn.functional.batch_norm(
            input_17,
            l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_17 = l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        batch_norm_13 = torch.nn.functional.batch_norm(
            input_16,
            l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_16 = l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_ = l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_ = l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_1_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_ = (None)
        x2_3 = input_18 + batch_norm_13
        input_18 = batch_norm_13 = None
        input_19 = torch.conv2d(
            x_4,
            l_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            16,
        )
        x_4 = l_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_0_parameters_weight_ = (None)
        input_20 = torch.nn.functional.batch_norm(
            input_19,
            l_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_19 = l_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_1_parameters_weight_ = l_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_1_parameters_bias_ = (None)
        input_21 = torch.conv2d(
            input_20,
            l_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_20 = l_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_2_parameters_weight_ = (None)
        input_22 = torch.nn.functional.batch_norm(
            input_21,
            l_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_3_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_3_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_3_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_21 = l_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_3_buffers_running_mean_ = l_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_3_buffers_running_var_ = l_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_3_parameters_weight_ = l_self_modules_blocks_modules_1_modules_0_modules_shortcut_modules_3_parameters_bias_ = (None)
        x2_3 += input_22
        x_8 = x2_3
        x2_3 = input_22 = None
        input_23 = torch.conv2d(
            x_8,
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
        input_24 = torch.nn.functional.batch_norm(
            input_23,
            l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_23 = l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_25 = torch.nn.functional.relu(input_24, inplace=True)
        input_24 = None
        input_26 = torch.conv2d(
            input_25,
            l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            40,
        )
        l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_27 = torch.nn.functional.batch_norm(
            input_26,
            l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_26 = l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        batch_norm_18 = torch.nn.functional.batch_norm(
            input_25,
            l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_25 = l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_2_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_ = (None)
        x2_4 = input_27 + batch_norm_18
        input_27 = batch_norm_18 = None
        x_9 = torch.nn.functional.relu(x2_4, inplace=False)
        x2_4 = None
        input_28 = torch.conv2d(
            x_9,
            l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_9 = l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_ = (None)
        input_29 = torch.nn.functional.batch_norm(
            input_28,
            l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_28 = l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_30 = torch.conv2d(
            input_29,
            l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            28,
        )
        l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_31 = torch.nn.functional.batch_norm(
            input_30,
            l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_30 = l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        batch_norm_21 = torch.nn.functional.batch_norm(
            input_29,
            l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_29 = l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_2_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_ = (None)
        x2_5 = input_31 + batch_norm_21
        input_31 = batch_norm_21 = None
        x2_5 += x_8
        x_10 = x2_5
        x2_5 = x_8 = None
        input_32 = torch.conv2d(
            x_10,
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
        input_33 = torch.nn.functional.batch_norm(
            input_32,
            l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_32 = l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_34 = torch.nn.functional.relu(input_33, inplace=True)
        input_33 = None
        input_35 = torch.conv2d(
            input_34,
            l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            40,
        )
        l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_36 = torch.nn.functional.batch_norm(
            input_35,
            l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_35 = l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        batch_norm_24 = torch.nn.functional.batch_norm(
            input_34,
            l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_34 = l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_3_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_ = (None)
        x2_6 = input_36 + batch_norm_24
        input_36 = batch_norm_24 = None
        x_11 = torch.nn.functional.relu(x2_6, inplace=False)
        x2_6 = None
        x_12 = torch.conv2d(
            x_11,
            l_self_modules_blocks_modules_3_modules_0_modules_conv_dw_parameters_weight_,
            None,
            (2, 2),
            (2, 2),
            (1, 1),
            40,
        )
        x_11 = (
            l_self_modules_blocks_modules_3_modules_0_modules_conv_dw_parameters_weight_
        ) = None
        x_13 = torch.nn.functional.batch_norm(
            x_12,
            l_self_modules_blocks_modules_3_modules_0_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_12 = l_self_modules_blocks_modules_3_modules_0_modules_bn_dw_buffers_running_mean_ = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn_dw_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn_dw_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn_dw_parameters_bias_
        ) = None
        x_se = x_13.mean((2, 3), keepdim=True)
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
        x_14 = x_13 * hardsigmoid
        x_13 = hardsigmoid = None
        input_37 = torch.conv2d(
            x_14,
            l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_14 = l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_ = (None)
        input_38 = torch.nn.functional.batch_norm(
            input_37,
            l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_37 = l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_39 = torch.conv2d(
            input_38,
            l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            44,
        )
        l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_40 = torch.nn.functional.batch_norm(
            input_39,
            l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_39 = l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        batch_norm_28 = torch.nn.functional.batch_norm(
            input_38,
            l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_38 = l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_3_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_ = (None)
        x2_7 = input_40 + batch_norm_28
        input_40 = batch_norm_28 = None
        input_41 = torch.conv2d(
            x_10,
            l_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_0_parameters_weight_,
            None,
            (2, 2),
            (2, 2),
            (1, 1),
            28,
        )
        x_10 = l_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_0_parameters_weight_ = (None)
        input_42 = torch.nn.functional.batch_norm(
            input_41,
            l_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_41 = l_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_1_parameters_weight_ = l_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_1_parameters_bias_ = (None)
        input_43 = torch.conv2d(
            input_42,
            l_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_42 = l_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_2_parameters_weight_ = (None)
        input_44 = torch.nn.functional.batch_norm(
            input_43,
            l_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_3_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_3_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_3_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_43 = l_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_3_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_3_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_3_parameters_weight_ = l_self_modules_blocks_modules_3_modules_0_modules_shortcut_modules_3_parameters_bias_ = (None)
        x2_7 += input_44
        x_15 = x2_7
        x2_7 = input_44 = None
        input_45 = torch.conv2d(
            x_15,
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
        input_46 = torch.nn.functional.batch_norm(
            input_45,
            l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_45 = l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_47 = torch.nn.functional.relu(input_46, inplace=True)
        input_46 = None
        input_48 = torch.conv2d(
            input_47,
            l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            68,
        )
        l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_49 = torch.nn.functional.batch_norm(
            input_48,
            l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_48 = l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        batch_norm_33 = torch.nn.functional.batch_norm(
            input_47,
            l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_47 = l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_ = l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_ = l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_4_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_ = (None)
        x2_8 = input_49 + batch_norm_33
        input_49 = batch_norm_33 = None
        x_16 = torch.nn.functional.relu(x2_8, inplace=False)
        x2_8 = None
        x_se_4 = x_16.mean((2, 3), keepdim=True)
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
        x_17 = x_16 * hardsigmoid_1
        x_16 = hardsigmoid_1 = None
        input_50 = torch.conv2d(
            x_17,
            l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_17 = l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_ = (None)
        input_51 = torch.nn.functional.batch_norm(
            input_50,
            l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_50 = l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_52 = torch.conv2d(
            input_51,
            l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            44,
        )
        l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_53 = torch.nn.functional.batch_norm(
            input_52,
            l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_52 = l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        batch_norm_36 = torch.nn.functional.batch_norm(
            input_51,
            l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_51 = l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_ = l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_ = l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_4_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_ = (None)
        x2_9 = input_53 + batch_norm_36
        input_53 = batch_norm_36 = None
        x2_9 += x_15
        x_18 = x2_9
        x2_9 = x_15 = None
        input_54 = torch.conv2d(
            x_18,
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
        input_55 = torch.nn.functional.batch_norm(
            input_54,
            l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_54 = l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_56 = torch.nn.functional.relu(input_55, inplace=True)
        input_55 = None
        input_57 = torch.conv2d(
            input_56,
            l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            132,
        )
        l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_58 = torch.nn.functional.batch_norm(
            input_57,
            l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_57 = l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        batch_norm_39 = torch.nn.functional.batch_norm(
            input_56,
            l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_56 = l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_ = l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_ = l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_5_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_ = (None)
        x2_10 = input_58 + batch_norm_39
        input_58 = batch_norm_39 = None
        x_19 = torch.nn.functional.relu(x2_10, inplace=False)
        x2_10 = None
        x_20 = torch.conv2d(
            x_19,
            l_self_modules_blocks_modules_5_modules_0_modules_conv_dw_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            132,
        )
        x_19 = (
            l_self_modules_blocks_modules_5_modules_0_modules_conv_dw_parameters_weight_
        ) = None
        x_21 = torch.nn.functional.batch_norm(
            x_20,
            l_self_modules_blocks_modules_5_modules_0_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_20 = l_self_modules_blocks_modules_5_modules_0_modules_bn_dw_buffers_running_mean_ = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn_dw_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn_dw_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn_dw_parameters_bias_
        ) = None
        input_59 = torch.conv2d(
            x_21,
            l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_21 = l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_ = (None)
        input_60 = torch.nn.functional.batch_norm(
            input_59,
            l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_59 = l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_61 = torch.conv2d(
            input_60,
            l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            88,
        )
        l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_62 = torch.nn.functional.batch_norm(
            input_61,
            l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_61 = l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        batch_norm_43 = torch.nn.functional.batch_norm(
            input_60,
            l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_60 = l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_ = l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_ = l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_5_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_ = (None)
        x2_11 = input_62 + batch_norm_43
        input_62 = batch_norm_43 = None
        input_63 = torch.conv2d(
            x_18,
            l_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            44,
        )
        x_18 = l_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_0_parameters_weight_ = (None)
        input_64 = torch.nn.functional.batch_norm(
            input_63,
            l_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_63 = l_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_1_parameters_weight_ = l_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_1_parameters_bias_ = (None)
        input_65 = torch.conv2d(
            input_64,
            l_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_64 = l_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_2_parameters_weight_ = (None)
        input_66 = torch.nn.functional.batch_norm(
            input_65,
            l_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_65 = l_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_3_buffers_running_mean_ = l_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_3_buffers_running_var_ = l_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_3_parameters_weight_ = l_self_modules_blocks_modules_5_modules_0_modules_shortcut_modules_3_parameters_bias_ = (None)
        x2_11 += input_66
        x_22 = x2_11
        x2_11 = input_66 = None
        input_67 = torch.conv2d(
            x_22,
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
        input_68 = torch.nn.functional.batch_norm(
            input_67,
            l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_67 = l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_69 = torch.nn.functional.relu(input_68, inplace=True)
        input_68 = None
        input_70 = torch.conv2d(
            input_69,
            l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            112,
        )
        l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_71 = torch.nn.functional.batch_norm(
            input_70,
            l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_70 = l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        batch_norm_48 = torch.nn.functional.batch_norm(
            input_69,
            l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_69 = l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_6_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_ = (None)
        x2_12 = input_71 + batch_norm_48
        input_71 = batch_norm_48 = None
        x_23 = torch.nn.functional.relu(x2_12, inplace=False)
        x2_12 = None
        input_72 = torch.conv2d(
            x_23,
            l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_23 = l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_ = (None)
        input_73 = torch.nn.functional.batch_norm(
            input_72,
            l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_72 = l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_74 = torch.conv2d(
            input_73,
            l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            88,
        )
        l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_75 = torch.nn.functional.batch_norm(
            input_74,
            l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_74 = l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        batch_norm_51 = torch.nn.functional.batch_norm(
            input_73,
            l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_73 = l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_6_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_ = (None)
        x2_13 = input_75 + batch_norm_51
        input_75 = batch_norm_51 = None
        x2_13 += x_22
        x_24 = x2_13
        x2_13 = x_22 = None
        input_76 = torch.conv2d(
            x_24,
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
        input_77 = torch.nn.functional.batch_norm(
            input_76,
            l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_76 = l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_78 = torch.nn.functional.relu(input_77, inplace=True)
        input_77 = None
        input_79 = torch.conv2d(
            input_78,
            l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            132,
        )
        l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_80 = torch.nn.functional.batch_norm(
            input_79,
            l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_79 = l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        batch_norm_54 = torch.nn.functional.batch_norm(
            input_78,
            l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_78 = l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_6_modules_1_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_ = (None)
        x2_14 = input_80 + batch_norm_54
        input_80 = batch_norm_54 = None
        x_25 = torch.nn.functional.relu(x2_14, inplace=False)
        x2_14 = None
        input_81 = torch.conv2d(
            x_25,
            l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_25 = l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_ = (None)
        input_82 = torch.nn.functional.batch_norm(
            input_81,
            l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_81 = l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_83 = torch.conv2d(
            input_82,
            l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            88,
        )
        l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_84 = torch.nn.functional.batch_norm(
            input_83,
            l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_83 = l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        batch_norm_57 = torch.nn.functional.batch_norm(
            input_82,
            l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_82 = l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_6_modules_1_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_ = (None)
        x2_15 = input_84 + batch_norm_57
        input_84 = batch_norm_57 = None
        x2_15 += x_24
        x_26 = x2_15
        x2_15 = x_24 = None
        input_85 = torch.conv2d(
            x_26,
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
        input_86 = torch.nn.functional.batch_norm(
            input_85,
            l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_85 = l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_87 = torch.nn.functional.relu(input_86, inplace=True)
        input_86 = None
        input_88 = torch.conv2d(
            input_87,
            l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            132,
        )
        l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_89 = torch.nn.functional.batch_norm(
            input_88,
            l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_88 = l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        batch_norm_60 = torch.nn.functional.batch_norm(
            input_87,
            l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_87 = l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_6_modules_2_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_ = (None)
        x2_16 = input_89 + batch_norm_60
        input_89 = batch_norm_60 = None
        x_27 = torch.nn.functional.relu(x2_16, inplace=False)
        x2_16 = None
        input_90 = torch.conv2d(
            x_27,
            l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_27 = l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_ = (None)
        input_91 = torch.nn.functional.batch_norm(
            input_90,
            l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_90 = l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_92 = torch.conv2d(
            input_91,
            l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            88,
        )
        l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_93 = torch.nn.functional.batch_norm(
            input_92,
            l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_92 = l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        batch_norm_63 = torch.nn.functional.batch_norm(
            input_91,
            l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_91 = l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_6_modules_2_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_ = (None)
        x2_17 = input_93 + batch_norm_63
        input_93 = batch_norm_63 = None
        x2_17 += x_26
        x_28 = x2_17
        x2_17 = x_26 = None
        input_94 = torch.conv2d(
            x_28,
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
        input_95 = torch.nn.functional.batch_norm(
            input_94,
            l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_94 = l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_96 = torch.nn.functional.relu(input_95, inplace=True)
        input_95 = None
        input_97 = torch.conv2d(
            input_96,
            l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            268,
        )
        l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_98 = torch.nn.functional.batch_norm(
            input_97,
            l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_97 = l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        batch_norm_66 = torch.nn.functional.batch_norm(
            input_96,
            l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_96 = l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_6_modules_3_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_ = (None)
        x2_18 = input_98 + batch_norm_66
        input_98 = batch_norm_66 = None
        x_29 = torch.nn.functional.relu(x2_18, inplace=False)
        x2_18 = None
        x_se_8 = x_29.mean((2, 3), keepdim=True)
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
        x_30 = x_29 * hardsigmoid_2
        x_29 = hardsigmoid_2 = None
        input_99 = torch.conv2d(
            x_30,
            l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_30 = l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_ = (None)
        input_100 = torch.nn.functional.batch_norm(
            input_99,
            l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_99 = l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_101 = torch.conv2d(
            input_100,
            l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            124,
        )
        l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_102 = torch.nn.functional.batch_norm(
            input_101,
            l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_101 = l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        batch_norm_69 = torch.nn.functional.batch_norm(
            input_100,
            l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_100 = l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_6_modules_3_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_ = (None)
        x2_19 = input_102 + batch_norm_69
        input_102 = batch_norm_69 = None
        input_103 = torch.conv2d(
            x_28,
            l_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            88,
        )
        x_28 = l_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_0_parameters_weight_ = (None)
        input_104 = torch.nn.functional.batch_norm(
            input_103,
            l_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_103 = l_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_1_parameters_weight_ = l_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_1_parameters_bias_ = (None)
        input_105 = torch.conv2d(
            input_104,
            l_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_104 = l_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_2_parameters_weight_ = (None)
        input_106 = torch.nn.functional.batch_norm(
            input_105,
            l_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_3_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_3_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_3_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_105 = l_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_3_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_3_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_3_parameters_weight_ = l_self_modules_blocks_modules_6_modules_3_modules_shortcut_modules_3_parameters_bias_ = (None)
        x2_19 += input_106
        x_31 = x2_19
        x2_19 = input_106 = None
        input_107 = torch.conv2d(
            x_31,
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
        input_108 = torch.nn.functional.batch_norm(
            input_107,
            l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_107 = l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_109 = torch.nn.functional.relu(input_108, inplace=True)
        input_108 = None
        input_110 = torch.conv2d(
            input_109,
            l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            372,
        )
        l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_111 = torch.nn.functional.batch_norm(
            input_110,
            l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_110 = l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        batch_norm_74 = torch.nn.functional.batch_norm(
            input_109,
            l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_109 = l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_6_modules_4_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_ = (None)
        x2_20 = input_111 + batch_norm_74
        input_111 = batch_norm_74 = None
        x_32 = torch.nn.functional.relu(x2_20, inplace=False)
        x2_20 = None
        x_se_12 = x_32.mean((2, 3), keepdim=True)
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
        x_33 = x_32 * hardsigmoid_3
        x_32 = hardsigmoid_3 = None
        input_112 = torch.conv2d(
            x_33,
            l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_33 = l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_ = (None)
        input_113 = torch.nn.functional.batch_norm(
            input_112,
            l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_112 = l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_114 = torch.conv2d(
            input_113,
            l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            124,
        )
        l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_115 = torch.nn.functional.batch_norm(
            input_114,
            l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_114 = l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        batch_norm_77 = torch.nn.functional.batch_norm(
            input_113,
            l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_113 = l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_6_modules_4_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_ = (None)
        x2_21 = input_115 + batch_norm_77
        input_115 = batch_norm_77 = None
        x2_21 += x_31
        x_34 = x2_21
        x2_21 = x_31 = None
        input_116 = torch.conv2d(
            x_34,
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
        input_117 = torch.nn.functional.batch_norm(
            input_116,
            l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_116 = l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_118 = torch.nn.functional.relu(input_117, inplace=True)
        input_117 = None
        input_119 = torch.conv2d(
            input_118,
            l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            372,
        )
        l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_120 = torch.nn.functional.batch_norm(
            input_119,
            l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_119 = l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        batch_norm_80 = torch.nn.functional.batch_norm(
            input_118,
            l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_,
            l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_,
            l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_118 = l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_ = l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_ = l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_7_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_ = (None)
        x2_22 = input_120 + batch_norm_80
        input_120 = batch_norm_80 = None
        x_35 = torch.nn.functional.relu(x2_22, inplace=False)
        x2_22 = None
        x_36 = torch.conv2d(
            x_35,
            l_self_modules_blocks_modules_7_modules_0_modules_conv_dw_parameters_weight_,
            None,
            (2, 2),
            (2, 2),
            (1, 1),
            372,
        )
        x_35 = (
            l_self_modules_blocks_modules_7_modules_0_modules_conv_dw_parameters_weight_
        ) = None
        x_37 = torch.nn.functional.batch_norm(
            x_36,
            l_self_modules_blocks_modules_7_modules_0_modules_bn_dw_buffers_running_mean_,
            l_self_modules_blocks_modules_7_modules_0_modules_bn_dw_buffers_running_var_,
            l_self_modules_blocks_modules_7_modules_0_modules_bn_dw_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_0_modules_bn_dw_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_36 = l_self_modules_blocks_modules_7_modules_0_modules_bn_dw_buffers_running_mean_ = (
            l_self_modules_blocks_modules_7_modules_0_modules_bn_dw_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_7_modules_0_modules_bn_dw_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_7_modules_0_modules_bn_dw_parameters_bias_
        ) = None
        x_se_16 = x_37.mean((2, 3), keepdim=True)
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
        x_38 = x_37 * hardsigmoid_4
        x_37 = hardsigmoid_4 = None
        input_121 = torch.conv2d(
            x_38,
            l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_38 = l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_ = (None)
        input_122 = torch.nn.functional.batch_norm(
            input_121,
            l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_121 = l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_123 = torch.conv2d(
            input_122,
            l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            176,
        )
        l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_124 = torch.nn.functional.batch_norm(
            input_123,
            l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_123 = l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        batch_norm_84 = torch.nn.functional.batch_norm(
            input_122,
            l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_,
            l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_,
            l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_122 = l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_ = l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_ = l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_7_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_ = (None)
        x2_23 = input_124 + batch_norm_84
        input_124 = batch_norm_84 = None
        input_125 = torch.conv2d(
            x_34,
            l_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_0_parameters_weight_,
            None,
            (2, 2),
            (2, 2),
            (1, 1),
            124,
        )
        x_34 = l_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_0_parameters_weight_ = (None)
        input_126 = torch.nn.functional.batch_norm(
            input_125,
            l_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_125 = l_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_1_parameters_weight_ = l_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_1_parameters_bias_ = (None)
        input_127 = torch.conv2d(
            input_126,
            l_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_126 = l_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_2_parameters_weight_ = (None)
        input_128 = torch.nn.functional.batch_norm(
            input_127,
            l_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_3_buffers_running_mean_,
            l_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_3_buffers_running_var_,
            l_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_3_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_127 = l_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_3_buffers_running_mean_ = l_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_3_buffers_running_var_ = l_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_3_parameters_weight_ = l_self_modules_blocks_modules_7_modules_0_modules_shortcut_modules_3_parameters_bias_ = (None)
        x2_23 += input_128
        x_39 = x2_23
        x2_23 = input_128 = None
        input_129 = torch.conv2d(
            x_39,
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
        input_130 = torch.nn.functional.batch_norm(
            input_129,
            l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_129 = l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_131 = torch.nn.functional.relu(input_130, inplace=True)
        input_130 = None
        input_132 = torch.conv2d(
            input_131,
            l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            532,
        )
        l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_133 = torch.nn.functional.batch_norm(
            input_132,
            l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_132 = l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        batch_norm_89 = torch.nn.functional.batch_norm(
            input_131,
            l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_131 = l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_ = l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_ = l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_8_modules_0_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_ = (None)
        x2_24 = input_133 + batch_norm_89
        input_133 = batch_norm_89 = None
        x_40 = torch.nn.functional.relu(x2_24, inplace=False)
        x2_24 = None
        input_134 = torch.conv2d(
            x_40,
            l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_40 = l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_ = (None)
        input_135 = torch.nn.functional.batch_norm(
            input_134,
            l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_134 = l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_136 = torch.conv2d(
            input_135,
            l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            176,
        )
        l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_137 = torch.nn.functional.batch_norm(
            input_136,
            l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_136 = l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        batch_norm_92 = torch.nn.functional.batch_norm(
            input_135,
            l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_135 = l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_ = l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_ = l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_8_modules_0_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_ = (None)
        x2_25 = input_137 + batch_norm_92
        input_137 = batch_norm_92 = None
        x2_25 += x_39
        x_41 = x2_25
        x2_25 = x_39 = None
        input_138 = torch.conv2d(
            x_41,
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
        input_139 = torch.nn.functional.batch_norm(
            input_138,
            l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_138 = l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_140 = torch.nn.functional.relu(input_139, inplace=True)
        input_139 = None
        input_141 = torch.conv2d(
            input_140,
            l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            532,
        )
        l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_142 = torch.nn.functional.batch_norm(
            input_141,
            l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_141 = l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        batch_norm_95 = torch.nn.functional.batch_norm(
            input_140,
            l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_140 = l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_ = l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_ = l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_8_modules_1_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_ = (None)
        x2_26 = input_142 + batch_norm_95
        input_142 = batch_norm_95 = None
        x_42 = torch.nn.functional.relu(x2_26, inplace=False)
        x2_26 = None
        x_se_20 = x_42.mean((2, 3), keepdim=True)
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
        x_43 = x_42 * hardsigmoid_5
        x_42 = hardsigmoid_5 = None
        input_143 = torch.conv2d(
            x_43,
            l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_43 = l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_ = (None)
        input_144 = torch.nn.functional.batch_norm(
            input_143,
            l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_143 = l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_145 = torch.conv2d(
            input_144,
            l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            176,
        )
        l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_146 = torch.nn.functional.batch_norm(
            input_145,
            l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_145 = l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        batch_norm_98 = torch.nn.functional.batch_norm(
            input_144,
            l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_144 = l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_ = l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_ = l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_8_modules_1_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_ = (None)
        x2_27 = input_146 + batch_norm_98
        input_146 = batch_norm_98 = None
        x2_27 += x_41
        x_44 = x2_27
        x2_27 = x_41 = None
        input_147 = torch.conv2d(
            x_44,
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
        input_148 = torch.nn.functional.batch_norm(
            input_147,
            l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_147 = l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_149 = torch.nn.functional.relu(input_148, inplace=True)
        input_148 = None
        input_150 = torch.conv2d(
            input_149,
            l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            532,
        )
        l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_151 = torch.nn.functional.batch_norm(
            input_150,
            l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_150 = l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        batch_norm_101 = torch.nn.functional.batch_norm(
            input_149,
            l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_149 = l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_ = l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_ = l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_8_modules_2_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_ = (None)
        x2_28 = input_151 + batch_norm_101
        input_151 = batch_norm_101 = None
        x_45 = torch.nn.functional.relu(x2_28, inplace=False)
        x2_28 = None
        input_152 = torch.conv2d(
            x_45,
            l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_45 = l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_ = (None)
        input_153 = torch.nn.functional.batch_norm(
            input_152,
            l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_152 = l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_154 = torch.conv2d(
            input_153,
            l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            176,
        )
        l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_155 = torch.nn.functional.batch_norm(
            input_154,
            l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_154 = l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        batch_norm_104 = torch.nn.functional.batch_norm(
            input_153,
            l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_153 = l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_ = l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_ = l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_8_modules_2_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_ = (None)
        x2_29 = input_155 + batch_norm_104
        input_155 = batch_norm_104 = None
        x2_29 += x_44
        x_46 = x2_29
        x2_29 = x_44 = None
        input_156 = torch.conv2d(
            x_46,
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
        input_157 = torch.nn.functional.batch_norm(
            input_156,
            l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_156 = l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_158 = torch.nn.functional.relu(input_157, inplace=True)
        input_157 = None
        input_159 = torch.conv2d(
            input_158,
            l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            532,
        )
        l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_160 = torch.nn.functional.batch_norm(
            input_159,
            l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_159 = l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        batch_norm_107 = torch.nn.functional.batch_norm(
            input_158,
            l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_158 = l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_mean_ = l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_fusion_bn_modules_0_buffers_running_var_ = l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_fusion_bn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_8_modules_3_modules_ghost1_modules_fusion_bn_modules_0_parameters_bias_ = (None)
        x2_30 = input_160 + batch_norm_107
        input_160 = batch_norm_107 = None
        x_47 = torch.nn.functional.relu(x2_30, inplace=False)
        x2_30 = None
        x_se_24 = x_47.mean((2, 3), keepdim=True)
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
        x_48 = x_47 * hardsigmoid_6
        x_47 = hardsigmoid_6 = None
        input_161 = torch.conv2d(
            x_48,
            l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_48 = l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_primary_conv_modules_0_parameters_weight_ = (None)
        input_162 = torch.nn.functional.batch_norm(
            input_161,
            l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_161 = l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_primary_conv_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_primary_conv_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_primary_conv_modules_1_parameters_weight_ = l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_primary_conv_modules_1_parameters_bias_ = (None)
        input_163 = torch.conv2d(
            input_162,
            l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            176,
        )
        l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_cheap_operation_modules_0_parameters_weight_ = (
            None
        )
        input_164 = torch.nn.functional.batch_norm(
            input_163,
            l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_163 = l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_mean_ = l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_cheap_operation_modules_1_buffers_running_var_ = l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_cheap_operation_modules_1_parameters_weight_ = l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_cheap_operation_modules_1_parameters_bias_ = (None)
        batch_norm_110 = torch.nn.functional.batch_norm(
            input_162,
            l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_162 = l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_mean_ = l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_fusion_bn_modules_0_buffers_running_var_ = l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_fusion_bn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_8_modules_3_modules_ghost2_modules_fusion_bn_modules_0_parameters_bias_ = (None)
        x2_31 = input_164 + batch_norm_110
        input_164 = batch_norm_110 = None
        x2_31 += x_46
        x_49 = x2_31
        x2_31 = x_46 = None
        x_50 = torch.conv2d(
            x_49,
            l_self_modules_blocks_modules_9_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_49 = (
            l_self_modules_blocks_modules_9_modules_0_modules_conv_parameters_weight_
        ) = None
        x_51 = torch.nn.functional.batch_norm(
            x_50,
            l_self_modules_blocks_modules_9_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_9_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_9_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_50 = (
            l_self_modules_blocks_modules_9_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_9_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_9_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_9_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_52 = torch.nn.functional.relu(x_51, inplace=True)
        x_51 = None
        x_53 = torch.nn.functional.adaptive_avg_pool2d(x_52, 1)
        x_52 = None
        x_54 = torch.conv2d(
            x_53,
            l_self_modules_conv_head_parameters_weight_,
            l_self_modules_conv_head_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_53 = (
            l_self_modules_conv_head_parameters_weight_
        ) = l_self_modules_conv_head_parameters_bias_ = None
        x_55 = torch.nn.functional.relu(x_54, inplace=True)
        x_54 = None
        x_56 = x_55.flatten(1, -1)
        x_55 = None
        x_57 = torch.nn.functional.dropout(x_56, p=0.2, training=False)
        x_56 = None
        x_58 = torch._C._nn.linear(
            x_57,
            l_self_modules_classifier_parameters_weight_,
            l_self_modules_classifier_parameters_bias_,
        )
        x_57 = (
            l_self_modules_classifier_parameters_weight_
        ) = l_self_modules_classifier_parameters_bias_ = None
        return (x_58,)
