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
        L_self_modules_blocks_modules_0_modules_0_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_0_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_0_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_0_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_0_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_0_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_0_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_0_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_0_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_1_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_2_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_1_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_2_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_3_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_0_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_0_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_0_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_0_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_0_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_0_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_0_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_conv_head_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_blocks_modules_0_modules_0_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_0_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_0_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_0_modules_0_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_0_modules_0_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_0_modules_0_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_0_modules_0_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_0_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_0_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_0_modules_0_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_0_modules_0_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_0_modules_0_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_0_modules_0_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_0_modules_0_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_0_modules_0_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_0_modules_0_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_0_modules_0_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_0_modules_0_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_0_modules_0_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_0_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_0_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_0_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_0_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_0_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_0_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_1_modules_0_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_1_modules_0_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_1_modules_0_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_1_modules_0_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_1_modules_0_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_1_modules_0_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_1_modules_0_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_1_modules_0_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_1_modules_0_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_1_modules_0_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_1_modules_0_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_1_modules_0_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_1_modules_0_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_1_modules_0_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_0_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_0_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_0_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_1_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_1_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_1_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_1_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_1_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_1_modules_1_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_1_modules_1_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_1_modules_1_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_1_modules_1_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_1_modules_1_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_1_modules_1_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_1_modules_1_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_1_modules_1_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_1_modules_1_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_1_modules_1_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_1_modules_1_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_1_modules_1_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_1_modules_1_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_1_modules_1_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_1_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_1_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_1_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_0_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_0_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_0_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_0_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_2_modules_0_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_2_modules_0_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_2_modules_0_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_2_modules_0_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_2_modules_0_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_2_modules_0_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_2_modules_0_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_0_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_0_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_0_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_1_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_1_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_1_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_1_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_2_modules_1_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_2_modules_1_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_2_modules_1_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_2_modules_1_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_2_modules_1_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_2_modules_1_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_2_modules_1_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_1_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_1_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_1_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_0_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_0_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_0_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_0_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_3_modules_0_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_3_modules_0_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_1_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_1_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_1_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_1_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_3_modules_1_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_3_modules_1_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_2_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_2_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_2_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_2_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_3_modules_2_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_3_modules_2_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_0_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_0_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_0_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_0_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_4_modules_0_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_4_modules_0_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_1_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_1_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_1_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_1_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_4_modules_1_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_4_modules_1_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_2_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_2_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_2_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_2_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_4_modules_2_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_4_modules_2_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_0_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_0_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_0_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_0_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_5_modules_0_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_5_modules_0_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_1_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_1_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_1_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_1_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_5_modules_1_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_5_modules_1_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_2_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_2_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_2_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_2_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_5_modules_2_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_5_modules_2_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_3_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_3_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_3_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_3_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_5_modules_3_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_5_modules_3_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_6_modules_0_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_0_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_0_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_6_modules_0_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_6_modules_0_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_6_modules_0_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_6_modules_0_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_0_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_0_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_6_modules_0_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_6_modules_0_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_0_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_0_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_6_modules_0_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_6_modules_0_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_6_modules_0_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_6_modules_0_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_0_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_0_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_6_modules_0_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_6_modules_0_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_6_modules_0_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_6_modules_0_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_6_modules_0_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_6_modules_0_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_6_modules_0_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_6_modules_0_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_6_modules_0_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_6_modules_0_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_6_modules_0_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_6_modules_0_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_6_modules_0_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_6_modules_0_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_6_modules_0_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_6_modules_0_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_0_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_0_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_6_modules_0_modules_bn3_parameters_bias_
        )
        l_self_modules_conv_head_parameters_weight_ = (
            L_self_modules_conv_head_parameters_weight_
        )
        l_self_modules_bn2_buffers_running_mean_ = (
            L_self_modules_bn2_buffers_running_mean_
        )
        l_self_modules_bn2_buffers_running_var_ = (
            L_self_modules_bn2_buffers_running_var_
        )
        l_self_modules_bn2_parameters_weight_ = L_self_modules_bn2_parameters_weight_
        l_self_modules_bn2_parameters_bias_ = L_self_modules_bn2_parameters_bias_
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
        x_2 = torch.nn.functional.silu(x_1, inplace=True)
        x_1 = None
        x_3 = torch.conv2d(
            x_2,
            l_self_modules_blocks_modules_0_modules_0_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_2 = (
            l_self_modules_blocks_modules_0_modules_0_modules_conv_dw_parameters_weight_
        ) = None
        x_4 = torch.nn.functional.batch_norm(
            x_3,
            l_self_modules_blocks_modules_0_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_0_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_0_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_3 = (
            l_self_modules_blocks_modules_0_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_0_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_0_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_5 = torch.nn.functional.silu(x_4, inplace=True)
        x_4 = None
        x_se = x_5.mean((2, 3), keepdim=True)
        x_se_1 = torch.conv2d(
            x_se,
            l_self_modules_blocks_modules_0_modules_0_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_0_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se = l_self_modules_blocks_modules_0_modules_0_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_0_modules_0_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_2 = torch.nn.functional.silu(x_se_1, inplace=True)
        x_se_1 = None
        x_se_3 = torch.conv2d(
            x_se_2,
            l_self_modules_blocks_modules_0_modules_0_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_0_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_2 = l_self_modules_blocks_modules_0_modules_0_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_0_modules_0_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid = torch.sigmoid(x_se_3)
        x_se_3 = None
        x_6 = x_5 * sigmoid
        x_5 = sigmoid = None
        x_7 = torch.conv2d(
            x_6,
            l_self_modules_blocks_modules_0_modules_0_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_6 = (
            l_self_modules_blocks_modules_0_modules_0_modules_conv_pw_parameters_weight_
        ) = None
        x_8 = torch.nn.functional.batch_norm(
            x_7,
            l_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_7 = (
            l_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_bias_
        ) = None
        x_9 = torch.conv2d(
            x_8,
            l_self_modules_blocks_modules_1_modules_0_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_8 = (
            l_self_modules_blocks_modules_1_modules_0_modules_conv_pw_parameters_weight_
        ) = None
        x_10 = torch.nn.functional.batch_norm(
            x_9,
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_9 = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_11 = torch.nn.functional.silu(x_10, inplace=True)
        x_10 = None
        x_12 = torch.conv2d(
            x_11,
            l_self_modules_blocks_modules_1_modules_0_modules_conv_dw_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            96,
        )
        x_11 = (
            l_self_modules_blocks_modules_1_modules_0_modules_conv_dw_parameters_weight_
        ) = None
        x_13 = torch.nn.functional.batch_norm(
            x_12,
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_12 = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_bias_
        ) = None
        x_14 = torch.nn.functional.silu(x_13, inplace=True)
        x_13 = None
        x_se_4 = x_14.mean((2, 3), keepdim=True)
        x_se_5 = torch.conv2d(
            x_se_4,
            l_self_modules_blocks_modules_1_modules_0_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_4 = l_self_modules_blocks_modules_1_modules_0_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_1_modules_0_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_6 = torch.nn.functional.silu(x_se_5, inplace=True)
        x_se_5 = None
        x_se_7 = torch.conv2d(
            x_se_6,
            l_self_modules_blocks_modules_1_modules_0_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_6 = l_self_modules_blocks_modules_1_modules_0_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_1_modules_0_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_1 = torch.sigmoid(x_se_7)
        x_se_7 = None
        x_15 = x_14 * sigmoid_1
        x_14 = sigmoid_1 = None
        x_16 = torch.conv2d(
            x_15,
            l_self_modules_blocks_modules_1_modules_0_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_15 = l_self_modules_blocks_modules_1_modules_0_modules_conv_pwl_parameters_weight_ = (None)
        x_17 = torch.nn.functional.batch_norm(
            x_16,
            l_self_modules_blocks_modules_1_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_16 = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn3_parameters_bias_
        ) = None
        x_18 = torch.conv2d(
            x_17,
            l_self_modules_blocks_modules_1_modules_1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_1_modules_1_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_19 = torch.nn.functional.batch_norm(
            x_18,
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_18 = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_bias_
        ) = None
        x_20 = torch.nn.functional.silu(x_19, inplace=True)
        x_19 = None
        x_21 = torch.conv2d(
            x_20,
            l_self_modules_blocks_modules_1_modules_1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            144,
        )
        x_20 = (
            l_self_modules_blocks_modules_1_modules_1_modules_conv_dw_parameters_weight_
        ) = None
        x_22 = torch.nn.functional.batch_norm(
            x_21,
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_21 = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_bias_
        ) = None
        x_23 = torch.nn.functional.silu(x_22, inplace=True)
        x_22 = None
        x_se_8 = x_23.mean((2, 3), keepdim=True)
        x_se_9 = torch.conv2d(
            x_se_8,
            l_self_modules_blocks_modules_1_modules_1_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_1_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_8 = l_self_modules_blocks_modules_1_modules_1_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_1_modules_1_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_10 = torch.nn.functional.silu(x_se_9, inplace=True)
        x_se_9 = None
        x_se_11 = torch.conv2d(
            x_se_10,
            l_self_modules_blocks_modules_1_modules_1_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_1_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_10 = l_self_modules_blocks_modules_1_modules_1_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_1_modules_1_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_2 = torch.sigmoid(x_se_11)
        x_se_11 = None
        x_24 = x_23 * sigmoid_2
        x_23 = sigmoid_2 = None
        x_25 = torch.conv2d(
            x_24,
            l_self_modules_blocks_modules_1_modules_1_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_24 = l_self_modules_blocks_modules_1_modules_1_modules_conv_pwl_parameters_weight_ = (None)
        x_26 = torch.nn.functional.batch_norm(
            x_25,
            l_self_modules_blocks_modules_1_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_25 = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn3_parameters_bias_
        ) = None
        x_27 = x_26 + x_17
        x_26 = x_17 = None
        x_28 = torch.conv2d(
            x_27,
            l_self_modules_blocks_modules_2_modules_0_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_27 = (
            l_self_modules_blocks_modules_2_modules_0_modules_conv_pw_parameters_weight_
        ) = None
        x_29 = torch.nn.functional.batch_norm(
            x_28,
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_28 = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_30 = torch.nn.functional.silu(x_29, inplace=True)
        x_29 = None
        x_31 = torch.conv2d(
            x_30,
            l_self_modules_blocks_modules_2_modules_0_modules_conv_dw_parameters_weight_,
            None,
            (2, 2),
            (2, 2),
            (1, 1),
            144,
        )
        x_30 = (
            l_self_modules_blocks_modules_2_modules_0_modules_conv_dw_parameters_weight_
        ) = None
        x_32 = torch.nn.functional.batch_norm(
            x_31,
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_31 = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_bias_
        ) = None
        x_33 = torch.nn.functional.silu(x_32, inplace=True)
        x_32 = None
        x_se_12 = x_33.mean((2, 3), keepdim=True)
        x_se_13 = torch.conv2d(
            x_se_12,
            l_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_12 = l_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_14 = torch.nn.functional.silu(x_se_13, inplace=True)
        x_se_13 = None
        x_se_15 = torch.conv2d(
            x_se_14,
            l_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_14 = l_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_3 = torch.sigmoid(x_se_15)
        x_se_15 = None
        x_34 = x_33 * sigmoid_3
        x_33 = sigmoid_3 = None
        x_35 = torch.conv2d(
            x_34,
            l_self_modules_blocks_modules_2_modules_0_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_34 = l_self_modules_blocks_modules_2_modules_0_modules_conv_pwl_parameters_weight_ = (None)
        x_36 = torch.nn.functional.batch_norm(
            x_35,
            l_self_modules_blocks_modules_2_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_35 = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn3_parameters_bias_
        ) = None
        x_37 = torch.conv2d(
            x_36,
            l_self_modules_blocks_modules_2_modules_1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_2_modules_1_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_38 = torch.nn.functional.batch_norm(
            x_37,
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_37 = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_bias_
        ) = None
        x_39 = torch.nn.functional.silu(x_38, inplace=True)
        x_38 = None
        x_40 = torch.conv2d(
            x_39,
            l_self_modules_blocks_modules_2_modules_1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            192,
        )
        x_39 = (
            l_self_modules_blocks_modules_2_modules_1_modules_conv_dw_parameters_weight_
        ) = None
        x_41 = torch.nn.functional.batch_norm(
            x_40,
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_40 = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_bias_
        ) = None
        x_42 = torch.nn.functional.silu(x_41, inplace=True)
        x_41 = None
        x_se_16 = x_42.mean((2, 3), keepdim=True)
        x_se_17 = torch.conv2d(
            x_se_16,
            l_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_16 = l_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_18 = torch.nn.functional.silu(x_se_17, inplace=True)
        x_se_17 = None
        x_se_19 = torch.conv2d(
            x_se_18,
            l_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_18 = l_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_4 = torch.sigmoid(x_se_19)
        x_se_19 = None
        x_43 = x_42 * sigmoid_4
        x_42 = sigmoid_4 = None
        x_44 = torch.conv2d(
            x_43,
            l_self_modules_blocks_modules_2_modules_1_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_43 = l_self_modules_blocks_modules_2_modules_1_modules_conv_pwl_parameters_weight_ = (None)
        x_45 = torch.nn.functional.batch_norm(
            x_44,
            l_self_modules_blocks_modules_2_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_44 = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn3_parameters_bias_
        ) = None
        x_46 = x_45 + x_36
        x_45 = x_36 = None
        x_47 = torch.conv2d(
            x_46,
            l_self_modules_blocks_modules_3_modules_0_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_46 = (
            l_self_modules_blocks_modules_3_modules_0_modules_conv_pw_parameters_weight_
        ) = None
        x_48 = torch.nn.functional.batch_norm(
            x_47,
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_47 = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_49 = torch.nn.functional.silu(x_48, inplace=True)
        x_48 = None
        x_50 = torch.conv2d(
            x_49,
            l_self_modules_blocks_modules_3_modules_0_modules_conv_dw_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            192,
        )
        x_49 = (
            l_self_modules_blocks_modules_3_modules_0_modules_conv_dw_parameters_weight_
        ) = None
        x_51 = torch.nn.functional.batch_norm(
            x_50,
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_50 = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_bias_
        ) = None
        x_52 = torch.nn.functional.silu(x_51, inplace=True)
        x_51 = None
        x_se_20 = x_52.mean((2, 3), keepdim=True)
        x_se_21 = torch.conv2d(
            x_se_20,
            l_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_20 = l_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_22 = torch.nn.functional.silu(x_se_21, inplace=True)
        x_se_21 = None
        x_se_23 = torch.conv2d(
            x_se_22,
            l_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_22 = l_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_5 = torch.sigmoid(x_se_23)
        x_se_23 = None
        x_53 = x_52 * sigmoid_5
        x_52 = sigmoid_5 = None
        x_54 = torch.conv2d(
            x_53,
            l_self_modules_blocks_modules_3_modules_0_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_53 = l_self_modules_blocks_modules_3_modules_0_modules_conv_pwl_parameters_weight_ = (None)
        x_55 = torch.nn.functional.batch_norm(
            x_54,
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_54 = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_bias_
        ) = None
        x_56 = torch.conv2d(
            x_55,
            l_self_modules_blocks_modules_3_modules_1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_3_modules_1_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_57 = torch.nn.functional.batch_norm(
            x_56,
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_56 = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_bias_
        ) = None
        x_58 = torch.nn.functional.silu(x_57, inplace=True)
        x_57 = None
        x_59 = torch.conv2d(
            x_58,
            l_self_modules_blocks_modules_3_modules_1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        x_58 = (
            l_self_modules_blocks_modules_3_modules_1_modules_conv_dw_parameters_weight_
        ) = None
        x_60 = torch.nn.functional.batch_norm(
            x_59,
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_59 = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_bias_
        ) = None
        x_61 = torch.nn.functional.silu(x_60, inplace=True)
        x_60 = None
        x_se_24 = x_61.mean((2, 3), keepdim=True)
        x_se_25 = torch.conv2d(
            x_se_24,
            l_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_24 = l_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_26 = torch.nn.functional.silu(x_se_25, inplace=True)
        x_se_25 = None
        x_se_27 = torch.conv2d(
            x_se_26,
            l_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_26 = l_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_6 = torch.sigmoid(x_se_27)
        x_se_27 = None
        x_62 = x_61 * sigmoid_6
        x_61 = sigmoid_6 = None
        x_63 = torch.conv2d(
            x_62,
            l_self_modules_blocks_modules_3_modules_1_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_62 = l_self_modules_blocks_modules_3_modules_1_modules_conv_pwl_parameters_weight_ = (None)
        x_64 = torch.nn.functional.batch_norm(
            x_63,
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_63 = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_bias_
        ) = None
        x_65 = x_64 + x_55
        x_64 = x_55 = None
        x_66 = torch.conv2d(
            x_65,
            l_self_modules_blocks_modules_3_modules_2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_3_modules_2_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_67 = torch.nn.functional.batch_norm(
            x_66,
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_66 = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_bias_
        ) = None
        x_68 = torch.nn.functional.silu(x_67, inplace=True)
        x_67 = None
        x_69 = torch.conv2d(
            x_68,
            l_self_modules_blocks_modules_3_modules_2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        x_68 = (
            l_self_modules_blocks_modules_3_modules_2_modules_conv_dw_parameters_weight_
        ) = None
        x_70 = torch.nn.functional.batch_norm(
            x_69,
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_69 = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_bias_
        ) = None
        x_71 = torch.nn.functional.silu(x_70, inplace=True)
        x_70 = None
        x_se_28 = x_71.mean((2, 3), keepdim=True)
        x_se_29 = torch.conv2d(
            x_se_28,
            l_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_28 = l_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_30 = torch.nn.functional.silu(x_se_29, inplace=True)
        x_se_29 = None
        x_se_31 = torch.conv2d(
            x_se_30,
            l_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_30 = l_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_7 = torch.sigmoid(x_se_31)
        x_se_31 = None
        x_72 = x_71 * sigmoid_7
        x_71 = sigmoid_7 = None
        x_73 = torch.conv2d(
            x_72,
            l_self_modules_blocks_modules_3_modules_2_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_72 = l_self_modules_blocks_modules_3_modules_2_modules_conv_pwl_parameters_weight_ = (None)
        x_74 = torch.nn.functional.batch_norm(
            x_73,
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_73 = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_bias_
        ) = None
        x_75 = x_74 + x_65
        x_74 = x_65 = None
        x_76 = torch.conv2d(
            x_75,
            l_self_modules_blocks_modules_4_modules_0_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_75 = (
            l_self_modules_blocks_modules_4_modules_0_modules_conv_pw_parameters_weight_
        ) = None
        x_77 = torch.nn.functional.batch_norm(
            x_76,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_76 = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_78 = torch.nn.functional.silu(x_77, inplace=True)
        x_77 = None
        x_79 = torch.conv2d(
            x_78,
            l_self_modules_blocks_modules_4_modules_0_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        x_78 = (
            l_self_modules_blocks_modules_4_modules_0_modules_conv_dw_parameters_weight_
        ) = None
        x_80 = torch.nn.functional.batch_norm(
            x_79,
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_79 = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_bias_
        ) = None
        x_81 = torch.nn.functional.silu(x_80, inplace=True)
        x_80 = None
        x_se_32 = x_81.mean((2, 3), keepdim=True)
        x_se_33 = torch.conv2d(
            x_se_32,
            l_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_32 = l_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_34 = torch.nn.functional.silu(x_se_33, inplace=True)
        x_se_33 = None
        x_se_35 = torch.conv2d(
            x_se_34,
            l_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_34 = l_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_8 = torch.sigmoid(x_se_35)
        x_se_35 = None
        x_82 = x_81 * sigmoid_8
        x_81 = sigmoid_8 = None
        x_83 = torch.conv2d(
            x_82,
            l_self_modules_blocks_modules_4_modules_0_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_82 = l_self_modules_blocks_modules_4_modules_0_modules_conv_pwl_parameters_weight_ = (None)
        x_84 = torch.nn.functional.batch_norm(
            x_83,
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_83 = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_bias_
        ) = None
        x_85 = torch.conv2d(
            x_84,
            l_self_modules_blocks_modules_4_modules_1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_4_modules_1_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_86 = torch.nn.functional.batch_norm(
            x_85,
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_85 = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_bias_
        ) = None
        x_87 = torch.nn.functional.silu(x_86, inplace=True)
        x_86 = None
        x_88 = torch.conv2d(
            x_87,
            l_self_modules_blocks_modules_4_modules_1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            528,
        )
        x_87 = (
            l_self_modules_blocks_modules_4_modules_1_modules_conv_dw_parameters_weight_
        ) = None
        x_89 = torch.nn.functional.batch_norm(
            x_88,
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_88 = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_bias_
        ) = None
        x_90 = torch.nn.functional.silu(x_89, inplace=True)
        x_89 = None
        x_se_36 = x_90.mean((2, 3), keepdim=True)
        x_se_37 = torch.conv2d(
            x_se_36,
            l_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_36 = l_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_38 = torch.nn.functional.silu(x_se_37, inplace=True)
        x_se_37 = None
        x_se_39 = torch.conv2d(
            x_se_38,
            l_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_38 = l_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_9 = torch.sigmoid(x_se_39)
        x_se_39 = None
        x_91 = x_90 * sigmoid_9
        x_90 = sigmoid_9 = None
        x_92 = torch.conv2d(
            x_91,
            l_self_modules_blocks_modules_4_modules_1_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_91 = l_self_modules_blocks_modules_4_modules_1_modules_conv_pwl_parameters_weight_ = (None)
        x_93 = torch.nn.functional.batch_norm(
            x_92,
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_92 = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_bias_
        ) = None
        x_94 = x_93 + x_84
        x_93 = x_84 = None
        x_95 = torch.conv2d(
            x_94,
            l_self_modules_blocks_modules_4_modules_2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_4_modules_2_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_96 = torch.nn.functional.batch_norm(
            x_95,
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_95 = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_bias_
        ) = None
        x_97 = torch.nn.functional.silu(x_96, inplace=True)
        x_96 = None
        x_98 = torch.conv2d(
            x_97,
            l_self_modules_blocks_modules_4_modules_2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            528,
        )
        x_97 = (
            l_self_modules_blocks_modules_4_modules_2_modules_conv_dw_parameters_weight_
        ) = None
        x_99 = torch.nn.functional.batch_norm(
            x_98,
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_98 = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_bias_
        ) = None
        x_100 = torch.nn.functional.silu(x_99, inplace=True)
        x_99 = None
        x_se_40 = x_100.mean((2, 3), keepdim=True)
        x_se_41 = torch.conv2d(
            x_se_40,
            l_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_40 = l_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_42 = torch.nn.functional.silu(x_se_41, inplace=True)
        x_se_41 = None
        x_se_43 = torch.conv2d(
            x_se_42,
            l_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_42 = l_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_10 = torch.sigmoid(x_se_43)
        x_se_43 = None
        x_101 = x_100 * sigmoid_10
        x_100 = sigmoid_10 = None
        x_102 = torch.conv2d(
            x_101,
            l_self_modules_blocks_modules_4_modules_2_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_101 = l_self_modules_blocks_modules_4_modules_2_modules_conv_pwl_parameters_weight_ = (None)
        x_103 = torch.nn.functional.batch_norm(
            x_102,
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_102 = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_bias_
        ) = None
        x_104 = x_103 + x_94
        x_103 = x_94 = None
        x_105 = torch.conv2d(
            x_104,
            l_self_modules_blocks_modules_5_modules_0_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_104 = (
            l_self_modules_blocks_modules_5_modules_0_modules_conv_pw_parameters_weight_
        ) = None
        x_106 = torch.nn.functional.batch_norm(
            x_105,
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_105 = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_107 = torch.nn.functional.silu(x_106, inplace=True)
        x_106 = None
        x_108 = torch.conv2d(
            x_107,
            l_self_modules_blocks_modules_5_modules_0_modules_conv_dw_parameters_weight_,
            None,
            (2, 2),
            (2, 2),
            (1, 1),
            528,
        )
        x_107 = (
            l_self_modules_blocks_modules_5_modules_0_modules_conv_dw_parameters_weight_
        ) = None
        x_109 = torch.nn.functional.batch_norm(
            x_108,
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_108 = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_bias_
        ) = None
        x_110 = torch.nn.functional.silu(x_109, inplace=True)
        x_109 = None
        x_se_44 = x_110.mean((2, 3), keepdim=True)
        x_se_45 = torch.conv2d(
            x_se_44,
            l_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_44 = l_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_46 = torch.nn.functional.silu(x_se_45, inplace=True)
        x_se_45 = None
        x_se_47 = torch.conv2d(
            x_se_46,
            l_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_46 = l_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_11 = torch.sigmoid(x_se_47)
        x_se_47 = None
        x_111 = x_110 * sigmoid_11
        x_110 = sigmoid_11 = None
        x_112 = torch.conv2d(
            x_111,
            l_self_modules_blocks_modules_5_modules_0_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_111 = l_self_modules_blocks_modules_5_modules_0_modules_conv_pwl_parameters_weight_ = (None)
        x_113 = torch.nn.functional.batch_norm(
            x_112,
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_112 = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_bias_
        ) = None
        x_114 = torch.conv2d(
            x_113,
            l_self_modules_blocks_modules_5_modules_1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_5_modules_1_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_115 = torch.nn.functional.batch_norm(
            x_114,
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_114 = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_bias_
        ) = None
        x_116 = torch.nn.functional.silu(x_115, inplace=True)
        x_115 = None
        x_117 = torch.conv2d(
            x_116,
            l_self_modules_blocks_modules_5_modules_1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            864,
        )
        x_116 = (
            l_self_modules_blocks_modules_5_modules_1_modules_conv_dw_parameters_weight_
        ) = None
        x_118 = torch.nn.functional.batch_norm(
            x_117,
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_117 = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_bias_
        ) = None
        x_119 = torch.nn.functional.silu(x_118, inplace=True)
        x_118 = None
        x_se_48 = x_119.mean((2, 3), keepdim=True)
        x_se_49 = torch.conv2d(
            x_se_48,
            l_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_48 = l_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_50 = torch.nn.functional.silu(x_se_49, inplace=True)
        x_se_49 = None
        x_se_51 = torch.conv2d(
            x_se_50,
            l_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_50 = l_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_12 = torch.sigmoid(x_se_51)
        x_se_51 = None
        x_120 = x_119 * sigmoid_12
        x_119 = sigmoid_12 = None
        x_121 = torch.conv2d(
            x_120,
            l_self_modules_blocks_modules_5_modules_1_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_120 = l_self_modules_blocks_modules_5_modules_1_modules_conv_pwl_parameters_weight_ = (None)
        x_122 = torch.nn.functional.batch_norm(
            x_121,
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_121 = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_bias_
        ) = None
        x_123 = x_122 + x_113
        x_122 = x_113 = None
        x_124 = torch.conv2d(
            x_123,
            l_self_modules_blocks_modules_5_modules_2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_5_modules_2_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_125 = torch.nn.functional.batch_norm(
            x_124,
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_124 = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_bias_
        ) = None
        x_126 = torch.nn.functional.silu(x_125, inplace=True)
        x_125 = None
        x_127 = torch.conv2d(
            x_126,
            l_self_modules_blocks_modules_5_modules_2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            864,
        )
        x_126 = (
            l_self_modules_blocks_modules_5_modules_2_modules_conv_dw_parameters_weight_
        ) = None
        x_128 = torch.nn.functional.batch_norm(
            x_127,
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_127 = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_bias_
        ) = None
        x_129 = torch.nn.functional.silu(x_128, inplace=True)
        x_128 = None
        x_se_52 = x_129.mean((2, 3), keepdim=True)
        x_se_53 = torch.conv2d(
            x_se_52,
            l_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_52 = l_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_54 = torch.nn.functional.silu(x_se_53, inplace=True)
        x_se_53 = None
        x_se_55 = torch.conv2d(
            x_se_54,
            l_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_54 = l_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_13 = torch.sigmoid(x_se_55)
        x_se_55 = None
        x_130 = x_129 * sigmoid_13
        x_129 = sigmoid_13 = None
        x_131 = torch.conv2d(
            x_130,
            l_self_modules_blocks_modules_5_modules_2_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_130 = l_self_modules_blocks_modules_5_modules_2_modules_conv_pwl_parameters_weight_ = (None)
        x_132 = torch.nn.functional.batch_norm(
            x_131,
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_131 = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_bias_
        ) = None
        x_133 = x_132 + x_123
        x_132 = x_123 = None
        x_134 = torch.conv2d(
            x_133,
            l_self_modules_blocks_modules_5_modules_3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_5_modules_3_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_135 = torch.nn.functional.batch_norm(
            x_134,
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_134 = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_bias_
        ) = None
        x_136 = torch.nn.functional.silu(x_135, inplace=True)
        x_135 = None
        x_137 = torch.conv2d(
            x_136,
            l_self_modules_blocks_modules_5_modules_3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            864,
        )
        x_136 = (
            l_self_modules_blocks_modules_5_modules_3_modules_conv_dw_parameters_weight_
        ) = None
        x_138 = torch.nn.functional.batch_norm(
            x_137,
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_137 = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_bias_
        ) = None
        x_139 = torch.nn.functional.silu(x_138, inplace=True)
        x_138 = None
        x_se_56 = x_139.mean((2, 3), keepdim=True)
        x_se_57 = torch.conv2d(
            x_se_56,
            l_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_56 = l_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_58 = torch.nn.functional.silu(x_se_57, inplace=True)
        x_se_57 = None
        x_se_59 = torch.conv2d(
            x_se_58,
            l_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_58 = l_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_14 = torch.sigmoid(x_se_59)
        x_se_59 = None
        x_140 = x_139 * sigmoid_14
        x_139 = sigmoid_14 = None
        x_141 = torch.conv2d(
            x_140,
            l_self_modules_blocks_modules_5_modules_3_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_140 = l_self_modules_blocks_modules_5_modules_3_modules_conv_pwl_parameters_weight_ = (None)
        x_142 = torch.nn.functional.batch_norm(
            x_141,
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_141 = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_bias_
        ) = None
        x_143 = x_142 + x_133
        x_142 = x_133 = None
        x_144 = torch.conv2d(
            x_143,
            l_self_modules_blocks_modules_6_modules_0_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_143 = (
            l_self_modules_blocks_modules_6_modules_0_modules_conv_pw_parameters_weight_
        ) = None
        x_145 = torch.nn.functional.batch_norm(
            x_144,
            l_self_modules_blocks_modules_6_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_144 = (
            l_self_modules_blocks_modules_6_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_6_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_6_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_146 = torch.nn.functional.silu(x_145, inplace=True)
        x_145 = None
        x_147 = torch.conv2d(
            x_146,
            l_self_modules_blocks_modules_6_modules_0_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            864,
        )
        x_146 = (
            l_self_modules_blocks_modules_6_modules_0_modules_conv_dw_parameters_weight_
        ) = None
        x_148 = torch.nn.functional.batch_norm(
            x_147,
            l_self_modules_blocks_modules_6_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_147 = (
            l_self_modules_blocks_modules_6_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_6_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_6_modules_0_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_0_modules_bn2_parameters_bias_
        ) = None
        x_149 = torch.nn.functional.silu(x_148, inplace=True)
        x_148 = None
        x_se_60 = x_149.mean((2, 3), keepdim=True)
        x_se_61 = torch.conv2d(
            x_se_60,
            l_self_modules_blocks_modules_6_modules_0_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_0_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_60 = l_self_modules_blocks_modules_6_modules_0_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_6_modules_0_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_62 = torch.nn.functional.silu(x_se_61, inplace=True)
        x_se_61 = None
        x_se_63 = torch.conv2d(
            x_se_62,
            l_self_modules_blocks_modules_6_modules_0_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_0_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_62 = l_self_modules_blocks_modules_6_modules_0_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_6_modules_0_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_15 = torch.sigmoid(x_se_63)
        x_se_63 = None
        x_150 = x_149 * sigmoid_15
        x_149 = sigmoid_15 = None
        x_151 = torch.conv2d(
            x_150,
            l_self_modules_blocks_modules_6_modules_0_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_150 = l_self_modules_blocks_modules_6_modules_0_modules_conv_pwl_parameters_weight_ = (None)
        x_152 = torch.nn.functional.batch_norm(
            x_151,
            l_self_modules_blocks_modules_6_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_151 = (
            l_self_modules_blocks_modules_6_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_6_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_6_modules_0_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_0_modules_bn3_parameters_bias_
        ) = None
        x_153 = torch.conv2d(
            x_152,
            l_self_modules_conv_head_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_152 = l_self_modules_conv_head_parameters_weight_ = None
        x_154 = torch.nn.functional.batch_norm(
            x_153,
            l_self_modules_bn2_buffers_running_mean_,
            l_self_modules_bn2_buffers_running_var_,
            l_self_modules_bn2_parameters_weight_,
            l_self_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_153 = (
            l_self_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_bn2_parameters_weight_
        ) = l_self_modules_bn2_parameters_bias_ = None
        x_155 = torch.nn.functional.silu(x_154, inplace=True)
        x_154 = None
        x_156 = torch.nn.functional.adaptive_avg_pool2d(x_155, 1)
        x_155 = None
        x_157 = x_156.flatten(1, -1)
        x_156 = None
        x_158 = torch._C._nn.linear(
            x_157,
            l_self_modules_classifier_parameters_weight_,
            l_self_modules_classifier_parameters_bias_,
        )
        x_157 = (
            l_self_modules_classifier_parameters_weight_
        ) = l_self_modules_classifier_parameters_bias_ = None
        return (x_158,)
