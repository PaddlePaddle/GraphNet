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
        L_self_modules_blocks_modules_0_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_conv_exp_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_conv_exp_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_2_modules_conv_exp_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_2_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_3_modules_conv_exp_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_3_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_3_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_3_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_3_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_3_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_3_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_3_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_3_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_3_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_conv_exp_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_conv_exp_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_conv_exp_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_conv_exp_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_3_modules_3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_4_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_4_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_4_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_4_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_4_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_4_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_4_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_4_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_4_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_4_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_4_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_5_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_5_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_5_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_5_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_5_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_5_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_4_modules_3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_3_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_4_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_4_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_4_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_4_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_4_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_4_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_4_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_4_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_4_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_4_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_4_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_4_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_4_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_4_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_4_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_5_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_5_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_5_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_5_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_5_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_5_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_5_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_5_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_5_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_5_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_5_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_5_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_5_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_5_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_5_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_6_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_6_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_6_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_6_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_6_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_6_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_6_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_6_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_6_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_6_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_6_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_6_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_6_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_6_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_6_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_7_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_7_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_7_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_7_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_7_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_7_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_7_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_7_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_7_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_7_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_7_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_7_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_7_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_7_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_7_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_7_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_7_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_7_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_7_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_8_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_8_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_8_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_8_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_8_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_8_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_8_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_8_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_8_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_8_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_8_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_8_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_8_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_8_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_8_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_8_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_8_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_8_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_8_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_5_modules_4_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_4_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_4_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_4_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_4_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_4_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_4_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_4_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_4_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_4_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_4_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_4_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_4_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_4_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_4_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_5_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_5_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_5_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_5_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_5_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_5_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_5_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_5_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_5_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_5_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_5_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_5_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_5_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_5_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_5_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_6_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_6_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_6_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_6_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_6_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_6_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_6_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_6_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_6_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_6_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_6_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_6_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_6_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_6_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_6_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_7_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_7_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_7_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_7_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_7_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_7_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_7_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_7_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_7_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_7_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_7_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_7_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_7_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_7_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_7_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_8_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_8_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_8_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_8_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_8_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_8_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_8_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_8_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_8_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_8_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_8_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_8_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_8_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_8_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_8_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_9_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_9_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_9_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_9_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_9_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_9_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_9_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_9_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_9_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_9_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_9_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_9_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_9_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_9_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_9_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_9_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_9_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_9_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_9_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_10_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_10_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_10_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_10_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_10_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_10_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_10_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_10_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_10_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_10_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_10_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_10_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_10_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_10_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_10_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_10_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_10_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_10_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_10_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_11_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_11_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_11_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_11_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_11_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_11_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_11_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_11_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_11_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_11_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_11_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_11_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_11_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_11_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_11_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_11_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_11_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_11_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_11_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_12_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_12_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_12_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_12_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_12_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_12_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_12_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_12_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_12_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_12_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_12_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_12_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_12_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_12_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_12_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_12_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_12_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_12_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_12_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_13_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_13_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_13_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_13_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_13_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_13_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_13_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_13_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_13_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_13_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_13_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_13_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_13_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_13_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_13_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_13_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_13_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_13_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_13_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_blocks_modules_0_modules_0_modules_conv_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_0_modules_conv_parameters_weight_
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
        l_self_modules_blocks_modules_0_modules_1_modules_conv_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_1_modules_conv_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_1_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_0_modules_1_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_0_modules_1_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_0_modules_1_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_0_modules_1_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_1_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_1_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_0_modules_1_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_0_modules_conv_exp_parameters_weight_ = L_self_modules_blocks_modules_1_modules_0_modules_conv_exp_parameters_weight_
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
        l_self_modules_blocks_modules_1_modules_0_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_1_modules_0_modules_conv_pwl_parameters_weight_
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
        l_self_modules_blocks_modules_1_modules_1_modules_conv_exp_parameters_weight_ = L_self_modules_blocks_modules_1_modules_1_modules_conv_exp_parameters_weight_
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
        l_self_modules_blocks_modules_1_modules_1_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_1_modules_1_modules_conv_pwl_parameters_weight_
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
        l_self_modules_blocks_modules_1_modules_2_modules_conv_exp_parameters_weight_ = L_self_modules_blocks_modules_1_modules_2_modules_conv_exp_parameters_weight_
        l_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_2_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_1_modules_2_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_3_modules_conv_exp_parameters_weight_ = L_self_modules_blocks_modules_1_modules_3_modules_conv_exp_parameters_weight_
        l_self_modules_blocks_modules_1_modules_3_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_1_modules_3_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_1_modules_3_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_1_modules_3_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_1_modules_3_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_3_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_3_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_3_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_3_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_1_modules_3_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_1_modules_3_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_1_modules_3_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_1_modules_3_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_1_modules_3_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_1_modules_3_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_3_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_3_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_3_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_0_modules_conv_exp_parameters_weight_ = L_self_modules_blocks_modules_2_modules_0_modules_conv_exp_parameters_weight_
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
        l_self_modules_blocks_modules_2_modules_0_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_2_modules_0_modules_conv_pwl_parameters_weight_
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
        l_self_modules_blocks_modules_2_modules_1_modules_conv_exp_parameters_weight_ = L_self_modules_blocks_modules_2_modules_1_modules_conv_exp_parameters_weight_
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
        l_self_modules_blocks_modules_2_modules_1_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_2_modules_1_modules_conv_pwl_parameters_weight_
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
        l_self_modules_blocks_modules_2_modules_2_modules_conv_exp_parameters_weight_ = L_self_modules_blocks_modules_2_modules_2_modules_conv_exp_parameters_weight_
        l_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_2_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_2_modules_2_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_3_modules_conv_exp_parameters_weight_ = L_self_modules_blocks_modules_2_modules_3_modules_conv_exp_parameters_weight_
        l_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_3_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_2_modules_3_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_bias_
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
        l_self_modules_blocks_modules_3_modules_3_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_3_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_3_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_3_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_3_modules_3_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_3_modules_3_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_4_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_4_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_4_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_4_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_4_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_3_modules_4_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_3_modules_4_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_3_modules_4_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_3_modules_4_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_3_modules_4_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_3_modules_4_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_3_modules_4_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_3_modules_4_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_3_modules_4_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_3_modules_4_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_3_modules_4_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_3_modules_4_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_3_modules_4_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_3_modules_4_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_4_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_4_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_4_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_5_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_5_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_5_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_3_modules_5_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_3_modules_5_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_3_modules_5_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_3_modules_5_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_5_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_5_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_5_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_5_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_5_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_5_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_3_modules_5_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_3_modules_5_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_3_modules_5_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_3_modules_5_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_5_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_5_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_5_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_5_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_3_modules_5_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_3_modules_5_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_3_modules_5_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_3_modules_5_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_3_modules_5_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_3_modules_5_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_3_modules_5_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_3_modules_5_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_3_modules_5_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_3_modules_5_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_3_modules_5_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_3_modules_5_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_3_modules_5_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_3_modules_5_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_5_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_5_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_5_modules_bn3_parameters_bias_
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
        l_self_modules_blocks_modules_4_modules_3_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_3_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_3_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_3_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_4_modules_3_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_4_modules_3_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_4_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_4_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_4_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_4_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_4_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_4_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_4_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_4_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_4_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_4_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_4_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_4_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_4_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_4_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_4_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_4_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_4_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_4_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_4_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_4_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_4_modules_4_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_4_modules_4_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_4_modules_4_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_4_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_4_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_4_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_4_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_4_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_4_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_4_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_5_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_5_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_5_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_5_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_5_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_5_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_5_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_5_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_5_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_5_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_5_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_5_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_5_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_5_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_5_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_5_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_5_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_5_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_5_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_5_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_4_modules_5_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_4_modules_5_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_4_modules_5_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_5_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_5_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_5_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_5_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_5_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_5_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_5_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_6_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_6_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_6_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_6_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_6_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_6_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_6_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_6_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_6_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_6_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_6_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_6_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_6_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_6_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_6_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_6_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_6_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_6_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_6_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_6_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_4_modules_6_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_4_modules_6_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_4_modules_6_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_6_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_6_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_6_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_6_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_6_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_6_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_6_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_7_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_7_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_7_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_7_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_7_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_7_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_7_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_7_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_7_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_7_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_7_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_7_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_7_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_7_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_7_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_7_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_7_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_7_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_7_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_7_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_7_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_4_modules_7_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_4_modules_7_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_4_modules_7_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_4_modules_7_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_4_modules_7_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_4_modules_7_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_4_modules_7_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_4_modules_7_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_4_modules_7_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_4_modules_7_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_7_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_7_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_7_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_7_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_7_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_7_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_7_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_8_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_8_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_8_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_8_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_8_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_8_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_8_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_8_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_8_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_8_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_8_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_8_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_8_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_8_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_8_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_8_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_8_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_8_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_8_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_8_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_8_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_4_modules_8_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_4_modules_8_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_4_modules_8_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_4_modules_8_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_4_modules_8_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_4_modules_8_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_4_modules_8_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_4_modules_8_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_4_modules_8_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_4_modules_8_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_8_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_8_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_8_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_8_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_8_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_8_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_8_modules_bn3_parameters_bias_
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
        l_self_modules_blocks_modules_5_modules_4_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_4_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_4_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_4_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_4_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_4_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_4_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_4_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_4_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_4_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_4_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_4_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_4_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_4_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_4_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_4_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_4_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_4_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_4_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_4_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_5_modules_4_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_5_modules_4_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_5_modules_4_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_4_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_4_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_4_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_4_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_4_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_4_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_4_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_5_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_5_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_5_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_5_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_5_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_5_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_5_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_5_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_5_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_5_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_5_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_5_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_5_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_5_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_5_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_5_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_5_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_5_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_5_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_5_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_5_modules_5_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_5_modules_5_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_5_modules_5_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_5_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_5_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_5_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_5_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_5_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_5_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_5_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_6_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_6_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_6_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_6_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_6_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_6_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_6_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_6_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_6_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_6_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_6_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_6_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_6_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_6_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_6_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_6_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_6_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_6_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_6_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_6_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_5_modules_6_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_5_modules_6_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_5_modules_6_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_6_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_6_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_6_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_6_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_6_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_6_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_6_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_7_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_7_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_7_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_7_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_7_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_7_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_7_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_7_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_7_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_7_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_7_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_7_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_7_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_7_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_7_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_7_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_7_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_7_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_7_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_7_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_5_modules_7_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_5_modules_7_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_5_modules_7_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_7_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_7_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_7_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_7_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_7_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_7_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_7_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_8_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_8_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_8_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_8_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_8_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_8_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_8_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_8_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_8_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_8_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_8_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_8_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_8_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_8_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_8_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_8_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_8_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_8_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_8_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_8_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_5_modules_8_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_5_modules_8_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_5_modules_8_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_8_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_8_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_8_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_8_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_8_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_8_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_8_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_9_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_9_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_9_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_9_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_9_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_9_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_9_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_9_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_9_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_9_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_9_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_9_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_9_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_9_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_9_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_9_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_9_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_9_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_9_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_9_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_9_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_5_modules_9_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_5_modules_9_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_5_modules_9_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_5_modules_9_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_5_modules_9_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_5_modules_9_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_5_modules_9_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_5_modules_9_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_5_modules_9_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_5_modules_9_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_9_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_9_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_9_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_9_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_9_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_9_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_9_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_10_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_5_modules_10_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_5_modules_10_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_10_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_10_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_10_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_10_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_10_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_10_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_10_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_10_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_5_modules_10_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_5_modules_10_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_10_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_10_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_10_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_10_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_10_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_10_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_10_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_10_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_5_modules_10_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_5_modules_10_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_5_modules_10_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_5_modules_10_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_5_modules_10_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_5_modules_10_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_5_modules_10_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_5_modules_10_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_5_modules_10_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_5_modules_10_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_10_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_10_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_10_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_10_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_10_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_10_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_10_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_11_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_5_modules_11_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_5_modules_11_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_11_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_11_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_11_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_11_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_11_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_11_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_11_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_11_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_5_modules_11_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_5_modules_11_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_11_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_11_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_11_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_11_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_11_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_11_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_11_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_11_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_5_modules_11_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_5_modules_11_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_5_modules_11_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_5_modules_11_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_5_modules_11_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_5_modules_11_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_5_modules_11_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_5_modules_11_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_5_modules_11_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_5_modules_11_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_11_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_11_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_11_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_11_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_11_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_11_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_11_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_12_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_5_modules_12_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_5_modules_12_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_12_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_12_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_12_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_12_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_12_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_12_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_12_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_12_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_5_modules_12_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_5_modules_12_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_12_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_12_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_12_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_12_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_12_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_12_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_12_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_12_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_5_modules_12_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_5_modules_12_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_5_modules_12_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_5_modules_12_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_5_modules_12_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_5_modules_12_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_5_modules_12_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_5_modules_12_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_5_modules_12_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_5_modules_12_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_12_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_12_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_12_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_12_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_12_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_12_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_12_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_13_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_5_modules_13_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_5_modules_13_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_13_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_13_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_13_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_13_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_13_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_13_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_13_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_13_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_5_modules_13_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_5_modules_13_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_13_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_13_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_13_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_13_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_13_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_13_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_13_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_13_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_5_modules_13_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_5_modules_13_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_5_modules_13_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_5_modules_13_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_5_modules_13_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_5_modules_13_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_5_modules_13_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_5_modules_13_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_5_modules_13_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_5_modules_13_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_13_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_13_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_13_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_13_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_13_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_13_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_13_modules_bn3_parameters_bias_
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
            l_self_modules_blocks_modules_0_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_0_modules_0_modules_conv_parameters_weight_ = None
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
        x_6 = x_5 + x_2
        x_5 = x_2 = None
        x_7 = torch.conv2d(
            x_6,
            l_self_modules_blocks_modules_0_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_0_modules_1_modules_conv_parameters_weight_ = None
        x_8 = torch.nn.functional.batch_norm(
            x_7,
            l_self_modules_blocks_modules_0_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_0_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_0_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_7 = (
            l_self_modules_blocks_modules_0_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_0_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_0_modules_1_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_1_modules_bn1_parameters_bias_
        ) = None
        x_9 = torch.nn.functional.silu(x_8, inplace=True)
        x_8 = None
        x_10 = x_9 + x_6
        x_9 = x_6 = None
        x_11 = torch.conv2d(
            x_10,
            l_self_modules_blocks_modules_1_modules_0_modules_conv_exp_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_10 = l_self_modules_blocks_modules_1_modules_0_modules_conv_exp_parameters_weight_ = (None)
        x_12 = torch.nn.functional.batch_norm(
            x_11,
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_11 = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_13 = torch.nn.functional.silu(x_12, inplace=True)
        x_12 = None
        x_14 = torch.conv2d(
            x_13,
            l_self_modules_blocks_modules_1_modules_0_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_13 = l_self_modules_blocks_modules_1_modules_0_modules_conv_pwl_parameters_weight_ = (None)
        x_15 = torch.nn.functional.batch_norm(
            x_14,
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_14 = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_bias_
        ) = None
        x_16 = torch.conv2d(
            x_15,
            l_self_modules_blocks_modules_1_modules_1_modules_conv_exp_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_1_modules_1_modules_conv_exp_parameters_weight_ = (
            None
        )
        x_17 = torch.nn.functional.batch_norm(
            x_16,
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_16 = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_bias_
        ) = None
        x_18 = torch.nn.functional.silu(x_17, inplace=True)
        x_17 = None
        x_19 = torch.conv2d(
            x_18,
            l_self_modules_blocks_modules_1_modules_1_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_18 = l_self_modules_blocks_modules_1_modules_1_modules_conv_pwl_parameters_weight_ = (None)
        x_20 = torch.nn.functional.batch_norm(
            x_19,
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_19 = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_bias_
        ) = None
        x_21 = x_20 + x_15
        x_20 = x_15 = None
        x_22 = torch.conv2d(
            x_21,
            l_self_modules_blocks_modules_1_modules_2_modules_conv_exp_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_1_modules_2_modules_conv_exp_parameters_weight_ = (
            None
        )
        x_23 = torch.nn.functional.batch_norm(
            x_22,
            l_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_22 = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_bias_
        ) = None
        x_24 = torch.nn.functional.silu(x_23, inplace=True)
        x_23 = None
        x_25 = torch.conv2d(
            x_24,
            l_self_modules_blocks_modules_1_modules_2_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_24 = l_self_modules_blocks_modules_1_modules_2_modules_conv_pwl_parameters_weight_ = (None)
        x_26 = torch.nn.functional.batch_norm(
            x_25,
            l_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_25 = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_bias_
        ) = None
        x_27 = x_26 + x_21
        x_26 = x_21 = None
        x_28 = torch.conv2d(
            x_27,
            l_self_modules_blocks_modules_1_modules_3_modules_conv_exp_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_1_modules_3_modules_conv_exp_parameters_weight_ = (
            None
        )
        x_29 = torch.nn.functional.batch_norm(
            x_28,
            l_self_modules_blocks_modules_1_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_28 = (
            l_self_modules_blocks_modules_1_modules_3_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_3_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_3_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_3_modules_bn1_parameters_bias_
        ) = None
        x_30 = torch.nn.functional.silu(x_29, inplace=True)
        x_29 = None
        x_31 = torch.conv2d(
            x_30,
            l_self_modules_blocks_modules_1_modules_3_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_30 = l_self_modules_blocks_modules_1_modules_3_modules_conv_pwl_parameters_weight_ = (None)
        x_32 = torch.nn.functional.batch_norm(
            x_31,
            l_self_modules_blocks_modules_1_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_31 = (
            l_self_modules_blocks_modules_1_modules_3_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_3_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_3_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_3_modules_bn2_parameters_bias_
        ) = None
        x_33 = x_32 + x_27
        x_32 = x_27 = None
        x_34 = torch.conv2d(
            x_33,
            l_self_modules_blocks_modules_2_modules_0_modules_conv_exp_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_33 = l_self_modules_blocks_modules_2_modules_0_modules_conv_exp_parameters_weight_ = (None)
        x_35 = torch.nn.functional.batch_norm(
            x_34,
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_34 = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_36 = torch.nn.functional.silu(x_35, inplace=True)
        x_35 = None
        x_37 = torch.conv2d(
            x_36,
            l_self_modules_blocks_modules_2_modules_0_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_36 = l_self_modules_blocks_modules_2_modules_0_modules_conv_pwl_parameters_weight_ = (None)
        x_38 = torch.nn.functional.batch_norm(
            x_37,
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_37 = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_bias_
        ) = None
        x_39 = torch.conv2d(
            x_38,
            l_self_modules_blocks_modules_2_modules_1_modules_conv_exp_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_2_modules_1_modules_conv_exp_parameters_weight_ = (
            None
        )
        x_40 = torch.nn.functional.batch_norm(
            x_39,
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_39 = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_bias_
        ) = None
        x_41 = torch.nn.functional.silu(x_40, inplace=True)
        x_40 = None
        x_42 = torch.conv2d(
            x_41,
            l_self_modules_blocks_modules_2_modules_1_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_41 = l_self_modules_blocks_modules_2_modules_1_modules_conv_pwl_parameters_weight_ = (None)
        x_43 = torch.nn.functional.batch_norm(
            x_42,
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_42 = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_bias_
        ) = None
        x_44 = x_43 + x_38
        x_43 = x_38 = None
        x_45 = torch.conv2d(
            x_44,
            l_self_modules_blocks_modules_2_modules_2_modules_conv_exp_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_2_modules_2_modules_conv_exp_parameters_weight_ = (
            None
        )
        x_46 = torch.nn.functional.batch_norm(
            x_45,
            l_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_45 = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_bias_
        ) = None
        x_47 = torch.nn.functional.silu(x_46, inplace=True)
        x_46 = None
        x_48 = torch.conv2d(
            x_47,
            l_self_modules_blocks_modules_2_modules_2_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_47 = l_self_modules_blocks_modules_2_modules_2_modules_conv_pwl_parameters_weight_ = (None)
        x_49 = torch.nn.functional.batch_norm(
            x_48,
            l_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_48 = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_bias_
        ) = None
        x_50 = x_49 + x_44
        x_49 = x_44 = None
        x_51 = torch.conv2d(
            x_50,
            l_self_modules_blocks_modules_2_modules_3_modules_conv_exp_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_2_modules_3_modules_conv_exp_parameters_weight_ = (
            None
        )
        x_52 = torch.nn.functional.batch_norm(
            x_51,
            l_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_51 = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_bias_
        ) = None
        x_53 = torch.nn.functional.silu(x_52, inplace=True)
        x_52 = None
        x_54 = torch.conv2d(
            x_53,
            l_self_modules_blocks_modules_2_modules_3_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_53 = l_self_modules_blocks_modules_2_modules_3_modules_conv_pwl_parameters_weight_ = (None)
        x_55 = torch.nn.functional.batch_norm(
            x_54,
            l_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_54 = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_bias_
        ) = None
        x_56 = x_55 + x_50
        x_55 = x_50 = None
        x_57 = torch.conv2d(
            x_56,
            l_self_modules_blocks_modules_3_modules_0_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_56 = (
            l_self_modules_blocks_modules_3_modules_0_modules_conv_pw_parameters_weight_
        ) = None
        x_58 = torch.nn.functional.batch_norm(
            x_57,
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_57 = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_59 = torch.nn.functional.silu(x_58, inplace=True)
        x_58 = None
        x_60 = torch.conv2d(
            x_59,
            l_self_modules_blocks_modules_3_modules_0_modules_conv_dw_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            192,
        )
        x_59 = (
            l_self_modules_blocks_modules_3_modules_0_modules_conv_dw_parameters_weight_
        ) = None
        x_61 = torch.nn.functional.batch_norm(
            x_60,
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_60 = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_bias_
        ) = None
        x_62 = torch.nn.functional.silu(x_61, inplace=True)
        x_61 = None
        x_se = x_62.mean((2, 3), keepdim=True)
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
        x_se_2 = torch.nn.functional.silu(x_se_1, inplace=True)
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
        sigmoid = torch.sigmoid(x_se_3)
        x_se_3 = None
        x_63 = x_62 * sigmoid
        x_62 = sigmoid = None
        x_64 = torch.conv2d(
            x_63,
            l_self_modules_blocks_modules_3_modules_0_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_63 = l_self_modules_blocks_modules_3_modules_0_modules_conv_pwl_parameters_weight_ = (None)
        x_65 = torch.nn.functional.batch_norm(
            x_64,
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_64 = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_bias_
        ) = None
        x_66 = torch.conv2d(
            x_65,
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
        x_67 = torch.nn.functional.batch_norm(
            x_66,
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_66 = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_bias_
        ) = None
        x_68 = torch.nn.functional.silu(x_67, inplace=True)
        x_67 = None
        x_69 = torch.conv2d(
            x_68,
            l_self_modules_blocks_modules_3_modules_1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            416,
        )
        x_68 = (
            l_self_modules_blocks_modules_3_modules_1_modules_conv_dw_parameters_weight_
        ) = None
        x_70 = torch.nn.functional.batch_norm(
            x_69,
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_69 = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_bias_
        ) = None
        x_71 = torch.nn.functional.silu(x_70, inplace=True)
        x_70 = None
        x_se_4 = x_71.mean((2, 3), keepdim=True)
        x_se_5 = torch.conv2d(
            x_se_4,
            l_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_4 = l_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_6 = torch.nn.functional.silu(x_se_5, inplace=True)
        x_se_5 = None
        x_se_7 = torch.conv2d(
            x_se_6,
            l_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_6 = l_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_1 = torch.sigmoid(x_se_7)
        x_se_7 = None
        x_72 = x_71 * sigmoid_1
        x_71 = sigmoid_1 = None
        x_73 = torch.conv2d(
            x_72,
            l_self_modules_blocks_modules_3_modules_1_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_72 = l_self_modules_blocks_modules_3_modules_1_modules_conv_pwl_parameters_weight_ = (None)
        x_74 = torch.nn.functional.batch_norm(
            x_73,
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_73 = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_bias_
        ) = None
        x_75 = x_74 + x_65
        x_74 = x_65 = None
        x_76 = torch.conv2d(
            x_75,
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
        x_77 = torch.nn.functional.batch_norm(
            x_76,
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_76 = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_bias_
        ) = None
        x_78 = torch.nn.functional.silu(x_77, inplace=True)
        x_77 = None
        x_79 = torch.conv2d(
            x_78,
            l_self_modules_blocks_modules_3_modules_2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            416,
        )
        x_78 = (
            l_self_modules_blocks_modules_3_modules_2_modules_conv_dw_parameters_weight_
        ) = None
        x_80 = torch.nn.functional.batch_norm(
            x_79,
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_79 = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_bias_
        ) = None
        x_81 = torch.nn.functional.silu(x_80, inplace=True)
        x_80 = None
        x_se_8 = x_81.mean((2, 3), keepdim=True)
        x_se_9 = torch.conv2d(
            x_se_8,
            l_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_8 = l_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_10 = torch.nn.functional.silu(x_se_9, inplace=True)
        x_se_9 = None
        x_se_11 = torch.conv2d(
            x_se_10,
            l_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_10 = l_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_2 = torch.sigmoid(x_se_11)
        x_se_11 = None
        x_82 = x_81 * sigmoid_2
        x_81 = sigmoid_2 = None
        x_83 = torch.conv2d(
            x_82,
            l_self_modules_blocks_modules_3_modules_2_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_82 = l_self_modules_blocks_modules_3_modules_2_modules_conv_pwl_parameters_weight_ = (None)
        x_84 = torch.nn.functional.batch_norm(
            x_83,
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_83 = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_bias_
        ) = None
        x_85 = x_84 + x_75
        x_84 = x_75 = None
        x_86 = torch.conv2d(
            x_85,
            l_self_modules_blocks_modules_3_modules_3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_3_modules_3_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_87 = torch.nn.functional.batch_norm(
            x_86,
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_86 = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_bias_
        ) = None
        x_88 = torch.nn.functional.silu(x_87, inplace=True)
        x_87 = None
        x_89 = torch.conv2d(
            x_88,
            l_self_modules_blocks_modules_3_modules_3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            416,
        )
        x_88 = (
            l_self_modules_blocks_modules_3_modules_3_modules_conv_dw_parameters_weight_
        ) = None
        x_90 = torch.nn.functional.batch_norm(
            x_89,
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_89 = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_bias_
        ) = None
        x_91 = torch.nn.functional.silu(x_90, inplace=True)
        x_90 = None
        x_se_12 = x_91.mean((2, 3), keepdim=True)
        x_se_13 = torch.conv2d(
            x_se_12,
            l_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_12 = l_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_14 = torch.nn.functional.silu(x_se_13, inplace=True)
        x_se_13 = None
        x_se_15 = torch.conv2d(
            x_se_14,
            l_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_14 = l_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_3 = torch.sigmoid(x_se_15)
        x_se_15 = None
        x_92 = x_91 * sigmoid_3
        x_91 = sigmoid_3 = None
        x_93 = torch.conv2d(
            x_92,
            l_self_modules_blocks_modules_3_modules_3_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_92 = l_self_modules_blocks_modules_3_modules_3_modules_conv_pwl_parameters_weight_ = (None)
        x_94 = torch.nn.functional.batch_norm(
            x_93,
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_93 = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_bias_
        ) = None
        x_95 = x_94 + x_85
        x_94 = x_85 = None
        x_96 = torch.conv2d(
            x_95,
            l_self_modules_blocks_modules_3_modules_4_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_3_modules_4_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_97 = torch.nn.functional.batch_norm(
            x_96,
            l_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_96 = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_bias_
        ) = None
        x_98 = torch.nn.functional.silu(x_97, inplace=True)
        x_97 = None
        x_99 = torch.conv2d(
            x_98,
            l_self_modules_blocks_modules_3_modules_4_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            416,
        )
        x_98 = (
            l_self_modules_blocks_modules_3_modules_4_modules_conv_dw_parameters_weight_
        ) = None
        x_100 = torch.nn.functional.batch_norm(
            x_99,
            l_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_99 = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_bias_
        ) = None
        x_101 = torch.nn.functional.silu(x_100, inplace=True)
        x_100 = None
        x_se_16 = x_101.mean((2, 3), keepdim=True)
        x_se_17 = torch.conv2d(
            x_se_16,
            l_self_modules_blocks_modules_3_modules_4_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_4_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_16 = l_self_modules_blocks_modules_3_modules_4_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_3_modules_4_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_18 = torch.nn.functional.silu(x_se_17, inplace=True)
        x_se_17 = None
        x_se_19 = torch.conv2d(
            x_se_18,
            l_self_modules_blocks_modules_3_modules_4_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_4_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_18 = l_self_modules_blocks_modules_3_modules_4_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_3_modules_4_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_4 = torch.sigmoid(x_se_19)
        x_se_19 = None
        x_102 = x_101 * sigmoid_4
        x_101 = sigmoid_4 = None
        x_103 = torch.conv2d(
            x_102,
            l_self_modules_blocks_modules_3_modules_4_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_102 = l_self_modules_blocks_modules_3_modules_4_modules_conv_pwl_parameters_weight_ = (None)
        x_104 = torch.nn.functional.batch_norm(
            x_103,
            l_self_modules_blocks_modules_3_modules_4_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_103 = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn3_parameters_bias_
        ) = None
        x_105 = x_104 + x_95
        x_104 = x_95 = None
        x_106 = torch.conv2d(
            x_105,
            l_self_modules_blocks_modules_3_modules_5_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_3_modules_5_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_107 = torch.nn.functional.batch_norm(
            x_106,
            l_self_modules_blocks_modules_3_modules_5_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_5_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_5_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_5_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_106 = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn1_parameters_bias_
        ) = None
        x_108 = torch.nn.functional.silu(x_107, inplace=True)
        x_107 = None
        x_109 = torch.conv2d(
            x_108,
            l_self_modules_blocks_modules_3_modules_5_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            416,
        )
        x_108 = (
            l_self_modules_blocks_modules_3_modules_5_modules_conv_dw_parameters_weight_
        ) = None
        x_110 = torch.nn.functional.batch_norm(
            x_109,
            l_self_modules_blocks_modules_3_modules_5_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_5_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_5_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_5_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_109 = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn2_parameters_bias_
        ) = None
        x_111 = torch.nn.functional.silu(x_110, inplace=True)
        x_110 = None
        x_se_20 = x_111.mean((2, 3), keepdim=True)
        x_se_21 = torch.conv2d(
            x_se_20,
            l_self_modules_blocks_modules_3_modules_5_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_5_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_20 = l_self_modules_blocks_modules_3_modules_5_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_3_modules_5_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_22 = torch.nn.functional.silu(x_se_21, inplace=True)
        x_se_21 = None
        x_se_23 = torch.conv2d(
            x_se_22,
            l_self_modules_blocks_modules_3_modules_5_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_5_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_22 = l_self_modules_blocks_modules_3_modules_5_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_3_modules_5_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_5 = torch.sigmoid(x_se_23)
        x_se_23 = None
        x_112 = x_111 * sigmoid_5
        x_111 = sigmoid_5 = None
        x_113 = torch.conv2d(
            x_112,
            l_self_modules_blocks_modules_3_modules_5_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_112 = l_self_modules_blocks_modules_3_modules_5_modules_conv_pwl_parameters_weight_ = (None)
        x_114 = torch.nn.functional.batch_norm(
            x_113,
            l_self_modules_blocks_modules_3_modules_5_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_5_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_5_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_5_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_113 = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn3_parameters_bias_
        ) = None
        x_115 = x_114 + x_105
        x_114 = x_105 = None
        x_116 = torch.conv2d(
            x_115,
            l_self_modules_blocks_modules_4_modules_0_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_115 = (
            l_self_modules_blocks_modules_4_modules_0_modules_conv_pw_parameters_weight_
        ) = None
        x_117 = torch.nn.functional.batch_norm(
            x_116,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_116 = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_118 = torch.nn.functional.silu(x_117, inplace=True)
        x_117 = None
        x_119 = torch.conv2d(
            x_118,
            l_self_modules_blocks_modules_4_modules_0_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            624,
        )
        x_118 = (
            l_self_modules_blocks_modules_4_modules_0_modules_conv_dw_parameters_weight_
        ) = None
        x_120 = torch.nn.functional.batch_norm(
            x_119,
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_119 = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_bias_
        ) = None
        x_121 = torch.nn.functional.silu(x_120, inplace=True)
        x_120 = None
        x_se_24 = x_121.mean((2, 3), keepdim=True)
        x_se_25 = torch.conv2d(
            x_se_24,
            l_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_24 = l_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_26 = torch.nn.functional.silu(x_se_25, inplace=True)
        x_se_25 = None
        x_se_27 = torch.conv2d(
            x_se_26,
            l_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_26 = l_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_6 = torch.sigmoid(x_se_27)
        x_se_27 = None
        x_122 = x_121 * sigmoid_6
        x_121 = sigmoid_6 = None
        x_123 = torch.conv2d(
            x_122,
            l_self_modules_blocks_modules_4_modules_0_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_122 = l_self_modules_blocks_modules_4_modules_0_modules_conv_pwl_parameters_weight_ = (None)
        x_124 = torch.nn.functional.batch_norm(
            x_123,
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_123 = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_bias_
        ) = None
        x_125 = torch.conv2d(
            x_124,
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
        x_126 = torch.nn.functional.batch_norm(
            x_125,
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_125 = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_bias_
        ) = None
        x_127 = torch.nn.functional.silu(x_126, inplace=True)
        x_126 = None
        x_128 = torch.conv2d(
            x_127,
            l_self_modules_blocks_modules_4_modules_1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        x_127 = (
            l_self_modules_blocks_modules_4_modules_1_modules_conv_dw_parameters_weight_
        ) = None
        x_129 = torch.nn.functional.batch_norm(
            x_128,
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_128 = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_bias_
        ) = None
        x_130 = torch.nn.functional.silu(x_129, inplace=True)
        x_129 = None
        x_se_28 = x_130.mean((2, 3), keepdim=True)
        x_se_29 = torch.conv2d(
            x_se_28,
            l_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_28 = l_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_30 = torch.nn.functional.silu(x_se_29, inplace=True)
        x_se_29 = None
        x_se_31 = torch.conv2d(
            x_se_30,
            l_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_30 = l_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_7 = torch.sigmoid(x_se_31)
        x_se_31 = None
        x_131 = x_130 * sigmoid_7
        x_130 = sigmoid_7 = None
        x_132 = torch.conv2d(
            x_131,
            l_self_modules_blocks_modules_4_modules_1_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_131 = l_self_modules_blocks_modules_4_modules_1_modules_conv_pwl_parameters_weight_ = (None)
        x_133 = torch.nn.functional.batch_norm(
            x_132,
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_132 = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_bias_
        ) = None
        x_134 = x_133 + x_124
        x_133 = x_124 = None
        x_135 = torch.conv2d(
            x_134,
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
        x_136 = torch.nn.functional.batch_norm(
            x_135,
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_135 = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_bias_
        ) = None
        x_137 = torch.nn.functional.silu(x_136, inplace=True)
        x_136 = None
        x_138 = torch.conv2d(
            x_137,
            l_self_modules_blocks_modules_4_modules_2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        x_137 = (
            l_self_modules_blocks_modules_4_modules_2_modules_conv_dw_parameters_weight_
        ) = None
        x_139 = torch.nn.functional.batch_norm(
            x_138,
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_138 = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_bias_
        ) = None
        x_140 = torch.nn.functional.silu(x_139, inplace=True)
        x_139 = None
        x_se_32 = x_140.mean((2, 3), keepdim=True)
        x_se_33 = torch.conv2d(
            x_se_32,
            l_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_32 = l_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_34 = torch.nn.functional.silu(x_se_33, inplace=True)
        x_se_33 = None
        x_se_35 = torch.conv2d(
            x_se_34,
            l_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_34 = l_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_8 = torch.sigmoid(x_se_35)
        x_se_35 = None
        x_141 = x_140 * sigmoid_8
        x_140 = sigmoid_8 = None
        x_142 = torch.conv2d(
            x_141,
            l_self_modules_blocks_modules_4_modules_2_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_141 = l_self_modules_blocks_modules_4_modules_2_modules_conv_pwl_parameters_weight_ = (None)
        x_143 = torch.nn.functional.batch_norm(
            x_142,
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_142 = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_bias_
        ) = None
        x_144 = x_143 + x_134
        x_143 = x_134 = None
        x_145 = torch.conv2d(
            x_144,
            l_self_modules_blocks_modules_4_modules_3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_4_modules_3_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_146 = torch.nn.functional.batch_norm(
            x_145,
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_145 = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_bias_
        ) = None
        x_147 = torch.nn.functional.silu(x_146, inplace=True)
        x_146 = None
        x_148 = torch.conv2d(
            x_147,
            l_self_modules_blocks_modules_4_modules_3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        x_147 = (
            l_self_modules_blocks_modules_4_modules_3_modules_conv_dw_parameters_weight_
        ) = None
        x_149 = torch.nn.functional.batch_norm(
            x_148,
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_148 = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_bias_
        ) = None
        x_150 = torch.nn.functional.silu(x_149, inplace=True)
        x_149 = None
        x_se_36 = x_150.mean((2, 3), keepdim=True)
        x_se_37 = torch.conv2d(
            x_se_36,
            l_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_36 = l_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_38 = torch.nn.functional.silu(x_se_37, inplace=True)
        x_se_37 = None
        x_se_39 = torch.conv2d(
            x_se_38,
            l_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_38 = l_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_9 = torch.sigmoid(x_se_39)
        x_se_39 = None
        x_151 = x_150 * sigmoid_9
        x_150 = sigmoid_9 = None
        x_152 = torch.conv2d(
            x_151,
            l_self_modules_blocks_modules_4_modules_3_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_151 = l_self_modules_blocks_modules_4_modules_3_modules_conv_pwl_parameters_weight_ = (None)
        x_153 = torch.nn.functional.batch_norm(
            x_152,
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_152 = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_bias_
        ) = None
        x_154 = x_153 + x_144
        x_153 = x_144 = None
        x_155 = torch.conv2d(
            x_154,
            l_self_modules_blocks_modules_4_modules_4_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_4_modules_4_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_156 = torch.nn.functional.batch_norm(
            x_155,
            l_self_modules_blocks_modules_4_modules_4_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_4_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_4_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_4_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_155 = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn1_parameters_bias_
        ) = None
        x_157 = torch.nn.functional.silu(x_156, inplace=True)
        x_156 = None
        x_158 = torch.conv2d(
            x_157,
            l_self_modules_blocks_modules_4_modules_4_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        x_157 = (
            l_self_modules_blocks_modules_4_modules_4_modules_conv_dw_parameters_weight_
        ) = None
        x_159 = torch.nn.functional.batch_norm(
            x_158,
            l_self_modules_blocks_modules_4_modules_4_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_4_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_4_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_4_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_158 = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn2_parameters_bias_
        ) = None
        x_160 = torch.nn.functional.silu(x_159, inplace=True)
        x_159 = None
        x_se_40 = x_160.mean((2, 3), keepdim=True)
        x_se_41 = torch.conv2d(
            x_se_40,
            l_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_40 = l_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_42 = torch.nn.functional.silu(x_se_41, inplace=True)
        x_se_41 = None
        x_se_43 = torch.conv2d(
            x_se_42,
            l_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_42 = l_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_10 = torch.sigmoid(x_se_43)
        x_se_43 = None
        x_161 = x_160 * sigmoid_10
        x_160 = sigmoid_10 = None
        x_162 = torch.conv2d(
            x_161,
            l_self_modules_blocks_modules_4_modules_4_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_161 = l_self_modules_blocks_modules_4_modules_4_modules_conv_pwl_parameters_weight_ = (None)
        x_163 = torch.nn.functional.batch_norm(
            x_162,
            l_self_modules_blocks_modules_4_modules_4_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_4_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_4_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_4_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_162 = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn3_parameters_bias_
        ) = None
        x_164 = x_163 + x_154
        x_163 = x_154 = None
        x_165 = torch.conv2d(
            x_164,
            l_self_modules_blocks_modules_4_modules_5_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_4_modules_5_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_166 = torch.nn.functional.batch_norm(
            x_165,
            l_self_modules_blocks_modules_4_modules_5_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_5_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_5_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_5_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_165 = (
            l_self_modules_blocks_modules_4_modules_5_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_5_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_5_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_5_modules_bn1_parameters_bias_
        ) = None
        x_167 = torch.nn.functional.silu(x_166, inplace=True)
        x_166 = None
        x_168 = torch.conv2d(
            x_167,
            l_self_modules_blocks_modules_4_modules_5_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        x_167 = (
            l_self_modules_blocks_modules_4_modules_5_modules_conv_dw_parameters_weight_
        ) = None
        x_169 = torch.nn.functional.batch_norm(
            x_168,
            l_self_modules_blocks_modules_4_modules_5_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_5_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_5_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_5_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_168 = (
            l_self_modules_blocks_modules_4_modules_5_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_5_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_5_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_5_modules_bn2_parameters_bias_
        ) = None
        x_170 = torch.nn.functional.silu(x_169, inplace=True)
        x_169 = None
        x_se_44 = x_170.mean((2, 3), keepdim=True)
        x_se_45 = torch.conv2d(
            x_se_44,
            l_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_44 = l_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_46 = torch.nn.functional.silu(x_se_45, inplace=True)
        x_se_45 = None
        x_se_47 = torch.conv2d(
            x_se_46,
            l_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_46 = l_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_11 = torch.sigmoid(x_se_47)
        x_se_47 = None
        x_171 = x_170 * sigmoid_11
        x_170 = sigmoid_11 = None
        x_172 = torch.conv2d(
            x_171,
            l_self_modules_blocks_modules_4_modules_5_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_171 = l_self_modules_blocks_modules_4_modules_5_modules_conv_pwl_parameters_weight_ = (None)
        x_173 = torch.nn.functional.batch_norm(
            x_172,
            l_self_modules_blocks_modules_4_modules_5_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_5_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_5_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_5_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_172 = (
            l_self_modules_blocks_modules_4_modules_5_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_5_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_5_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_5_modules_bn3_parameters_bias_
        ) = None
        x_174 = x_173 + x_164
        x_173 = x_164 = None
        x_175 = torch.conv2d(
            x_174,
            l_self_modules_blocks_modules_4_modules_6_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_4_modules_6_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_176 = torch.nn.functional.batch_norm(
            x_175,
            l_self_modules_blocks_modules_4_modules_6_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_6_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_6_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_6_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_175 = (
            l_self_modules_blocks_modules_4_modules_6_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_6_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_6_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_6_modules_bn1_parameters_bias_
        ) = None
        x_177 = torch.nn.functional.silu(x_176, inplace=True)
        x_176 = None
        x_178 = torch.conv2d(
            x_177,
            l_self_modules_blocks_modules_4_modules_6_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        x_177 = (
            l_self_modules_blocks_modules_4_modules_6_modules_conv_dw_parameters_weight_
        ) = None
        x_179 = torch.nn.functional.batch_norm(
            x_178,
            l_self_modules_blocks_modules_4_modules_6_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_6_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_6_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_6_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_178 = (
            l_self_modules_blocks_modules_4_modules_6_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_6_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_6_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_6_modules_bn2_parameters_bias_
        ) = None
        x_180 = torch.nn.functional.silu(x_179, inplace=True)
        x_179 = None
        x_se_48 = x_180.mean((2, 3), keepdim=True)
        x_se_49 = torch.conv2d(
            x_se_48,
            l_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_48 = l_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_50 = torch.nn.functional.silu(x_se_49, inplace=True)
        x_se_49 = None
        x_se_51 = torch.conv2d(
            x_se_50,
            l_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_50 = l_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_12 = torch.sigmoid(x_se_51)
        x_se_51 = None
        x_181 = x_180 * sigmoid_12
        x_180 = sigmoid_12 = None
        x_182 = torch.conv2d(
            x_181,
            l_self_modules_blocks_modules_4_modules_6_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_181 = l_self_modules_blocks_modules_4_modules_6_modules_conv_pwl_parameters_weight_ = (None)
        x_183 = torch.nn.functional.batch_norm(
            x_182,
            l_self_modules_blocks_modules_4_modules_6_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_6_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_6_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_6_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_182 = (
            l_self_modules_blocks_modules_4_modules_6_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_6_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_6_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_6_modules_bn3_parameters_bias_
        ) = None
        x_184 = x_183 + x_174
        x_183 = x_174 = None
        x_185 = torch.conv2d(
            x_184,
            l_self_modules_blocks_modules_4_modules_7_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_4_modules_7_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_186 = torch.nn.functional.batch_norm(
            x_185,
            l_self_modules_blocks_modules_4_modules_7_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_7_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_7_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_7_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_185 = (
            l_self_modules_blocks_modules_4_modules_7_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_7_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_7_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_7_modules_bn1_parameters_bias_
        ) = None
        x_187 = torch.nn.functional.silu(x_186, inplace=True)
        x_186 = None
        x_188 = torch.conv2d(
            x_187,
            l_self_modules_blocks_modules_4_modules_7_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        x_187 = (
            l_self_modules_blocks_modules_4_modules_7_modules_conv_dw_parameters_weight_
        ) = None
        x_189 = torch.nn.functional.batch_norm(
            x_188,
            l_self_modules_blocks_modules_4_modules_7_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_7_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_7_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_7_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_188 = (
            l_self_modules_blocks_modules_4_modules_7_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_7_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_7_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_7_modules_bn2_parameters_bias_
        ) = None
        x_190 = torch.nn.functional.silu(x_189, inplace=True)
        x_189 = None
        x_se_52 = x_190.mean((2, 3), keepdim=True)
        x_se_53 = torch.conv2d(
            x_se_52,
            l_self_modules_blocks_modules_4_modules_7_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_7_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_52 = l_self_modules_blocks_modules_4_modules_7_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_4_modules_7_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_54 = torch.nn.functional.silu(x_se_53, inplace=True)
        x_se_53 = None
        x_se_55 = torch.conv2d(
            x_se_54,
            l_self_modules_blocks_modules_4_modules_7_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_7_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_54 = l_self_modules_blocks_modules_4_modules_7_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_4_modules_7_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_13 = torch.sigmoid(x_se_55)
        x_se_55 = None
        x_191 = x_190 * sigmoid_13
        x_190 = sigmoid_13 = None
        x_192 = torch.conv2d(
            x_191,
            l_self_modules_blocks_modules_4_modules_7_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_191 = l_self_modules_blocks_modules_4_modules_7_modules_conv_pwl_parameters_weight_ = (None)
        x_193 = torch.nn.functional.batch_norm(
            x_192,
            l_self_modules_blocks_modules_4_modules_7_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_7_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_7_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_7_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_192 = (
            l_self_modules_blocks_modules_4_modules_7_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_7_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_7_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_7_modules_bn3_parameters_bias_
        ) = None
        x_194 = x_193 + x_184
        x_193 = x_184 = None
        x_195 = torch.conv2d(
            x_194,
            l_self_modules_blocks_modules_4_modules_8_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_4_modules_8_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_196 = torch.nn.functional.batch_norm(
            x_195,
            l_self_modules_blocks_modules_4_modules_8_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_8_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_8_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_8_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_195 = (
            l_self_modules_blocks_modules_4_modules_8_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_8_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_8_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_8_modules_bn1_parameters_bias_
        ) = None
        x_197 = torch.nn.functional.silu(x_196, inplace=True)
        x_196 = None
        x_198 = torch.conv2d(
            x_197,
            l_self_modules_blocks_modules_4_modules_8_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        x_197 = (
            l_self_modules_blocks_modules_4_modules_8_modules_conv_dw_parameters_weight_
        ) = None
        x_199 = torch.nn.functional.batch_norm(
            x_198,
            l_self_modules_blocks_modules_4_modules_8_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_8_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_8_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_8_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_198 = (
            l_self_modules_blocks_modules_4_modules_8_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_8_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_8_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_8_modules_bn2_parameters_bias_
        ) = None
        x_200 = torch.nn.functional.silu(x_199, inplace=True)
        x_199 = None
        x_se_56 = x_200.mean((2, 3), keepdim=True)
        x_se_57 = torch.conv2d(
            x_se_56,
            l_self_modules_blocks_modules_4_modules_8_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_8_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_56 = l_self_modules_blocks_modules_4_modules_8_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_4_modules_8_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_58 = torch.nn.functional.silu(x_se_57, inplace=True)
        x_se_57 = None
        x_se_59 = torch.conv2d(
            x_se_58,
            l_self_modules_blocks_modules_4_modules_8_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_8_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_58 = l_self_modules_blocks_modules_4_modules_8_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_4_modules_8_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_14 = torch.sigmoid(x_se_59)
        x_se_59 = None
        x_201 = x_200 * sigmoid_14
        x_200 = sigmoid_14 = None
        x_202 = torch.conv2d(
            x_201,
            l_self_modules_blocks_modules_4_modules_8_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_201 = l_self_modules_blocks_modules_4_modules_8_modules_conv_pwl_parameters_weight_ = (None)
        x_203 = torch.nn.functional.batch_norm(
            x_202,
            l_self_modules_blocks_modules_4_modules_8_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_8_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_8_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_8_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_202 = (
            l_self_modules_blocks_modules_4_modules_8_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_8_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_8_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_8_modules_bn3_parameters_bias_
        ) = None
        x_204 = x_203 + x_194
        x_203 = x_194 = None
        x_205 = torch.conv2d(
            x_204,
            l_self_modules_blocks_modules_5_modules_0_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_204 = (
            l_self_modules_blocks_modules_5_modules_0_modules_conv_pw_parameters_weight_
        ) = None
        x_206 = torch.nn.functional.batch_norm(
            x_205,
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_205 = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_207 = torch.nn.functional.silu(x_206, inplace=True)
        x_206 = None
        x_208 = torch.conv2d(
            x_207,
            l_self_modules_blocks_modules_5_modules_0_modules_conv_dw_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            768,
        )
        x_207 = (
            l_self_modules_blocks_modules_5_modules_0_modules_conv_dw_parameters_weight_
        ) = None
        x_209 = torch.nn.functional.batch_norm(
            x_208,
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_208 = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_bias_
        ) = None
        x_210 = torch.nn.functional.silu(x_209, inplace=True)
        x_209 = None
        x_se_60 = x_210.mean((2, 3), keepdim=True)
        x_se_61 = torch.conv2d(
            x_se_60,
            l_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_60 = l_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_62 = torch.nn.functional.silu(x_se_61, inplace=True)
        x_se_61 = None
        x_se_63 = torch.conv2d(
            x_se_62,
            l_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_62 = l_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_15 = torch.sigmoid(x_se_63)
        x_se_63 = None
        x_211 = x_210 * sigmoid_15
        x_210 = sigmoid_15 = None
        x_212 = torch.conv2d(
            x_211,
            l_self_modules_blocks_modules_5_modules_0_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_211 = l_self_modules_blocks_modules_5_modules_0_modules_conv_pwl_parameters_weight_ = (None)
        x_213 = torch.nn.functional.batch_norm(
            x_212,
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_212 = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_bias_
        ) = None
        x_214 = torch.conv2d(
            x_213,
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
        x_215 = torch.nn.functional.batch_norm(
            x_214,
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_214 = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_bias_
        ) = None
        x_216 = torch.nn.functional.silu(x_215, inplace=True)
        x_215 = None
        x_217 = torch.conv2d(
            x_216,
            l_self_modules_blocks_modules_5_modules_1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1248,
        )
        x_216 = (
            l_self_modules_blocks_modules_5_modules_1_modules_conv_dw_parameters_weight_
        ) = None
        x_218 = torch.nn.functional.batch_norm(
            x_217,
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_217 = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_bias_
        ) = None
        x_219 = torch.nn.functional.silu(x_218, inplace=True)
        x_218 = None
        x_se_64 = x_219.mean((2, 3), keepdim=True)
        x_se_65 = torch.conv2d(
            x_se_64,
            l_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_64 = l_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_66 = torch.nn.functional.silu(x_se_65, inplace=True)
        x_se_65 = None
        x_se_67 = torch.conv2d(
            x_se_66,
            l_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_66 = l_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_16 = torch.sigmoid(x_se_67)
        x_se_67 = None
        x_220 = x_219 * sigmoid_16
        x_219 = sigmoid_16 = None
        x_221 = torch.conv2d(
            x_220,
            l_self_modules_blocks_modules_5_modules_1_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_220 = l_self_modules_blocks_modules_5_modules_1_modules_conv_pwl_parameters_weight_ = (None)
        x_222 = torch.nn.functional.batch_norm(
            x_221,
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_221 = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_bias_
        ) = None
        x_223 = x_222 + x_213
        x_222 = x_213 = None
        x_224 = torch.conv2d(
            x_223,
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
        x_225 = torch.nn.functional.batch_norm(
            x_224,
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_224 = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_bias_
        ) = None
        x_226 = torch.nn.functional.silu(x_225, inplace=True)
        x_225 = None
        x_227 = torch.conv2d(
            x_226,
            l_self_modules_blocks_modules_5_modules_2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1248,
        )
        x_226 = (
            l_self_modules_blocks_modules_5_modules_2_modules_conv_dw_parameters_weight_
        ) = None
        x_228 = torch.nn.functional.batch_norm(
            x_227,
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_227 = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_bias_
        ) = None
        x_229 = torch.nn.functional.silu(x_228, inplace=True)
        x_228 = None
        x_se_68 = x_229.mean((2, 3), keepdim=True)
        x_se_69 = torch.conv2d(
            x_se_68,
            l_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_68 = l_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_70 = torch.nn.functional.silu(x_se_69, inplace=True)
        x_se_69 = None
        x_se_71 = torch.conv2d(
            x_se_70,
            l_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_70 = l_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_17 = torch.sigmoid(x_se_71)
        x_se_71 = None
        x_230 = x_229 * sigmoid_17
        x_229 = sigmoid_17 = None
        x_231 = torch.conv2d(
            x_230,
            l_self_modules_blocks_modules_5_modules_2_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_230 = l_self_modules_blocks_modules_5_modules_2_modules_conv_pwl_parameters_weight_ = (None)
        x_232 = torch.nn.functional.batch_norm(
            x_231,
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_231 = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_bias_
        ) = None
        x_233 = x_232 + x_223
        x_232 = x_223 = None
        x_234 = torch.conv2d(
            x_233,
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
        x_235 = torch.nn.functional.batch_norm(
            x_234,
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_234 = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_bias_
        ) = None
        x_236 = torch.nn.functional.silu(x_235, inplace=True)
        x_235 = None
        x_237 = torch.conv2d(
            x_236,
            l_self_modules_blocks_modules_5_modules_3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1248,
        )
        x_236 = (
            l_self_modules_blocks_modules_5_modules_3_modules_conv_dw_parameters_weight_
        ) = None
        x_238 = torch.nn.functional.batch_norm(
            x_237,
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_237 = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_bias_
        ) = None
        x_239 = torch.nn.functional.silu(x_238, inplace=True)
        x_238 = None
        x_se_72 = x_239.mean((2, 3), keepdim=True)
        x_se_73 = torch.conv2d(
            x_se_72,
            l_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_72 = l_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_74 = torch.nn.functional.silu(x_se_73, inplace=True)
        x_se_73 = None
        x_se_75 = torch.conv2d(
            x_se_74,
            l_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_74 = l_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_18 = torch.sigmoid(x_se_75)
        x_se_75 = None
        x_240 = x_239 * sigmoid_18
        x_239 = sigmoid_18 = None
        x_241 = torch.conv2d(
            x_240,
            l_self_modules_blocks_modules_5_modules_3_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_240 = l_self_modules_blocks_modules_5_modules_3_modules_conv_pwl_parameters_weight_ = (None)
        x_242 = torch.nn.functional.batch_norm(
            x_241,
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_241 = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_bias_
        ) = None
        x_243 = x_242 + x_233
        x_242 = x_233 = None
        x_244 = torch.conv2d(
            x_243,
            l_self_modules_blocks_modules_5_modules_4_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_5_modules_4_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_245 = torch.nn.functional.batch_norm(
            x_244,
            l_self_modules_blocks_modules_5_modules_4_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_4_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_4_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_4_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_244 = (
            l_self_modules_blocks_modules_5_modules_4_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_4_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_4_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_4_modules_bn1_parameters_bias_
        ) = None
        x_246 = torch.nn.functional.silu(x_245, inplace=True)
        x_245 = None
        x_247 = torch.conv2d(
            x_246,
            l_self_modules_blocks_modules_5_modules_4_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1248,
        )
        x_246 = (
            l_self_modules_blocks_modules_5_modules_4_modules_conv_dw_parameters_weight_
        ) = None
        x_248 = torch.nn.functional.batch_norm(
            x_247,
            l_self_modules_blocks_modules_5_modules_4_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_4_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_4_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_4_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_247 = (
            l_self_modules_blocks_modules_5_modules_4_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_4_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_4_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_4_modules_bn2_parameters_bias_
        ) = None
        x_249 = torch.nn.functional.silu(x_248, inplace=True)
        x_248 = None
        x_se_76 = x_249.mean((2, 3), keepdim=True)
        x_se_77 = torch.conv2d(
            x_se_76,
            l_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_76 = l_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_78 = torch.nn.functional.silu(x_se_77, inplace=True)
        x_se_77 = None
        x_se_79 = torch.conv2d(
            x_se_78,
            l_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_78 = l_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_19 = torch.sigmoid(x_se_79)
        x_se_79 = None
        x_250 = x_249 * sigmoid_19
        x_249 = sigmoid_19 = None
        x_251 = torch.conv2d(
            x_250,
            l_self_modules_blocks_modules_5_modules_4_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_250 = l_self_modules_blocks_modules_5_modules_4_modules_conv_pwl_parameters_weight_ = (None)
        x_252 = torch.nn.functional.batch_norm(
            x_251,
            l_self_modules_blocks_modules_5_modules_4_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_4_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_4_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_4_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_251 = (
            l_self_modules_blocks_modules_5_modules_4_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_4_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_4_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_4_modules_bn3_parameters_bias_
        ) = None
        x_253 = x_252 + x_243
        x_252 = x_243 = None
        x_254 = torch.conv2d(
            x_253,
            l_self_modules_blocks_modules_5_modules_5_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_5_modules_5_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_255 = torch.nn.functional.batch_norm(
            x_254,
            l_self_modules_blocks_modules_5_modules_5_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_5_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_5_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_5_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_254 = (
            l_self_modules_blocks_modules_5_modules_5_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_5_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_5_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_5_modules_bn1_parameters_bias_
        ) = None
        x_256 = torch.nn.functional.silu(x_255, inplace=True)
        x_255 = None
        x_257 = torch.conv2d(
            x_256,
            l_self_modules_blocks_modules_5_modules_5_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1248,
        )
        x_256 = (
            l_self_modules_blocks_modules_5_modules_5_modules_conv_dw_parameters_weight_
        ) = None
        x_258 = torch.nn.functional.batch_norm(
            x_257,
            l_self_modules_blocks_modules_5_modules_5_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_5_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_5_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_5_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_257 = (
            l_self_modules_blocks_modules_5_modules_5_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_5_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_5_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_5_modules_bn2_parameters_bias_
        ) = None
        x_259 = torch.nn.functional.silu(x_258, inplace=True)
        x_258 = None
        x_se_80 = x_259.mean((2, 3), keepdim=True)
        x_se_81 = torch.conv2d(
            x_se_80,
            l_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_80 = l_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_82 = torch.nn.functional.silu(x_se_81, inplace=True)
        x_se_81 = None
        x_se_83 = torch.conv2d(
            x_se_82,
            l_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_82 = l_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_20 = torch.sigmoid(x_se_83)
        x_se_83 = None
        x_260 = x_259 * sigmoid_20
        x_259 = sigmoid_20 = None
        x_261 = torch.conv2d(
            x_260,
            l_self_modules_blocks_modules_5_modules_5_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_260 = l_self_modules_blocks_modules_5_modules_5_modules_conv_pwl_parameters_weight_ = (None)
        x_262 = torch.nn.functional.batch_norm(
            x_261,
            l_self_modules_blocks_modules_5_modules_5_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_5_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_5_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_5_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_261 = (
            l_self_modules_blocks_modules_5_modules_5_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_5_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_5_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_5_modules_bn3_parameters_bias_
        ) = None
        x_263 = x_262 + x_253
        x_262 = x_253 = None
        x_264 = torch.conv2d(
            x_263,
            l_self_modules_blocks_modules_5_modules_6_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_5_modules_6_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_265 = torch.nn.functional.batch_norm(
            x_264,
            l_self_modules_blocks_modules_5_modules_6_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_6_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_6_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_6_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_264 = (
            l_self_modules_blocks_modules_5_modules_6_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_6_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_6_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_6_modules_bn1_parameters_bias_
        ) = None
        x_266 = torch.nn.functional.silu(x_265, inplace=True)
        x_265 = None
        x_267 = torch.conv2d(
            x_266,
            l_self_modules_blocks_modules_5_modules_6_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1248,
        )
        x_266 = (
            l_self_modules_blocks_modules_5_modules_6_modules_conv_dw_parameters_weight_
        ) = None
        x_268 = torch.nn.functional.batch_norm(
            x_267,
            l_self_modules_blocks_modules_5_modules_6_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_6_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_6_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_6_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_267 = (
            l_self_modules_blocks_modules_5_modules_6_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_6_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_6_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_6_modules_bn2_parameters_bias_
        ) = None
        x_269 = torch.nn.functional.silu(x_268, inplace=True)
        x_268 = None
        x_se_84 = x_269.mean((2, 3), keepdim=True)
        x_se_85 = torch.conv2d(
            x_se_84,
            l_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_84 = l_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_86 = torch.nn.functional.silu(x_se_85, inplace=True)
        x_se_85 = None
        x_se_87 = torch.conv2d(
            x_se_86,
            l_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_86 = l_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_21 = torch.sigmoid(x_se_87)
        x_se_87 = None
        x_270 = x_269 * sigmoid_21
        x_269 = sigmoid_21 = None
        x_271 = torch.conv2d(
            x_270,
            l_self_modules_blocks_modules_5_modules_6_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_270 = l_self_modules_blocks_modules_5_modules_6_modules_conv_pwl_parameters_weight_ = (None)
        x_272 = torch.nn.functional.batch_norm(
            x_271,
            l_self_modules_blocks_modules_5_modules_6_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_6_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_6_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_6_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_271 = (
            l_self_modules_blocks_modules_5_modules_6_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_6_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_6_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_6_modules_bn3_parameters_bias_
        ) = None
        x_273 = x_272 + x_263
        x_272 = x_263 = None
        x_274 = torch.conv2d(
            x_273,
            l_self_modules_blocks_modules_5_modules_7_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_5_modules_7_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_275 = torch.nn.functional.batch_norm(
            x_274,
            l_self_modules_blocks_modules_5_modules_7_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_7_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_7_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_7_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_274 = (
            l_self_modules_blocks_modules_5_modules_7_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_7_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_7_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_7_modules_bn1_parameters_bias_
        ) = None
        x_276 = torch.nn.functional.silu(x_275, inplace=True)
        x_275 = None
        x_277 = torch.conv2d(
            x_276,
            l_self_modules_blocks_modules_5_modules_7_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1248,
        )
        x_276 = (
            l_self_modules_blocks_modules_5_modules_7_modules_conv_dw_parameters_weight_
        ) = None
        x_278 = torch.nn.functional.batch_norm(
            x_277,
            l_self_modules_blocks_modules_5_modules_7_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_7_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_7_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_7_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_277 = (
            l_self_modules_blocks_modules_5_modules_7_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_7_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_7_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_7_modules_bn2_parameters_bias_
        ) = None
        x_279 = torch.nn.functional.silu(x_278, inplace=True)
        x_278 = None
        x_se_88 = x_279.mean((2, 3), keepdim=True)
        x_se_89 = torch.conv2d(
            x_se_88,
            l_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_88 = l_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_90 = torch.nn.functional.silu(x_se_89, inplace=True)
        x_se_89 = None
        x_se_91 = torch.conv2d(
            x_se_90,
            l_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_90 = l_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_22 = torch.sigmoid(x_se_91)
        x_se_91 = None
        x_280 = x_279 * sigmoid_22
        x_279 = sigmoid_22 = None
        x_281 = torch.conv2d(
            x_280,
            l_self_modules_blocks_modules_5_modules_7_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_280 = l_self_modules_blocks_modules_5_modules_7_modules_conv_pwl_parameters_weight_ = (None)
        x_282 = torch.nn.functional.batch_norm(
            x_281,
            l_self_modules_blocks_modules_5_modules_7_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_7_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_7_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_7_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_281 = (
            l_self_modules_blocks_modules_5_modules_7_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_7_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_7_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_7_modules_bn3_parameters_bias_
        ) = None
        x_283 = x_282 + x_273
        x_282 = x_273 = None
        x_284 = torch.conv2d(
            x_283,
            l_self_modules_blocks_modules_5_modules_8_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_5_modules_8_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_285 = torch.nn.functional.batch_norm(
            x_284,
            l_self_modules_blocks_modules_5_modules_8_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_8_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_8_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_8_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_284 = (
            l_self_modules_blocks_modules_5_modules_8_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_8_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_8_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_8_modules_bn1_parameters_bias_
        ) = None
        x_286 = torch.nn.functional.silu(x_285, inplace=True)
        x_285 = None
        x_287 = torch.conv2d(
            x_286,
            l_self_modules_blocks_modules_5_modules_8_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1248,
        )
        x_286 = (
            l_self_modules_blocks_modules_5_modules_8_modules_conv_dw_parameters_weight_
        ) = None
        x_288 = torch.nn.functional.batch_norm(
            x_287,
            l_self_modules_blocks_modules_5_modules_8_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_8_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_8_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_8_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_287 = (
            l_self_modules_blocks_modules_5_modules_8_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_8_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_8_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_8_modules_bn2_parameters_bias_
        ) = None
        x_289 = torch.nn.functional.silu(x_288, inplace=True)
        x_288 = None
        x_se_92 = x_289.mean((2, 3), keepdim=True)
        x_se_93 = torch.conv2d(
            x_se_92,
            l_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_92 = l_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_94 = torch.nn.functional.silu(x_se_93, inplace=True)
        x_se_93 = None
        x_se_95 = torch.conv2d(
            x_se_94,
            l_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_94 = l_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_23 = torch.sigmoid(x_se_95)
        x_se_95 = None
        x_290 = x_289 * sigmoid_23
        x_289 = sigmoid_23 = None
        x_291 = torch.conv2d(
            x_290,
            l_self_modules_blocks_modules_5_modules_8_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_290 = l_self_modules_blocks_modules_5_modules_8_modules_conv_pwl_parameters_weight_ = (None)
        x_292 = torch.nn.functional.batch_norm(
            x_291,
            l_self_modules_blocks_modules_5_modules_8_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_8_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_8_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_8_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_291 = (
            l_self_modules_blocks_modules_5_modules_8_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_8_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_8_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_8_modules_bn3_parameters_bias_
        ) = None
        x_293 = x_292 + x_283
        x_292 = x_283 = None
        x_294 = torch.conv2d(
            x_293,
            l_self_modules_blocks_modules_5_modules_9_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_5_modules_9_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_295 = torch.nn.functional.batch_norm(
            x_294,
            l_self_modules_blocks_modules_5_modules_9_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_9_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_9_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_9_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_294 = (
            l_self_modules_blocks_modules_5_modules_9_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_9_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_9_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_9_modules_bn1_parameters_bias_
        ) = None
        x_296 = torch.nn.functional.silu(x_295, inplace=True)
        x_295 = None
        x_297 = torch.conv2d(
            x_296,
            l_self_modules_blocks_modules_5_modules_9_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1248,
        )
        x_296 = (
            l_self_modules_blocks_modules_5_modules_9_modules_conv_dw_parameters_weight_
        ) = None
        x_298 = torch.nn.functional.batch_norm(
            x_297,
            l_self_modules_blocks_modules_5_modules_9_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_9_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_9_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_9_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_297 = (
            l_self_modules_blocks_modules_5_modules_9_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_9_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_9_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_9_modules_bn2_parameters_bias_
        ) = None
        x_299 = torch.nn.functional.silu(x_298, inplace=True)
        x_298 = None
        x_se_96 = x_299.mean((2, 3), keepdim=True)
        x_se_97 = torch.conv2d(
            x_se_96,
            l_self_modules_blocks_modules_5_modules_9_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_9_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_96 = l_self_modules_blocks_modules_5_modules_9_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_9_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_98 = torch.nn.functional.silu(x_se_97, inplace=True)
        x_se_97 = None
        x_se_99 = torch.conv2d(
            x_se_98,
            l_self_modules_blocks_modules_5_modules_9_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_9_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_98 = l_self_modules_blocks_modules_5_modules_9_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_9_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_24 = torch.sigmoid(x_se_99)
        x_se_99 = None
        x_300 = x_299 * sigmoid_24
        x_299 = sigmoid_24 = None
        x_301 = torch.conv2d(
            x_300,
            l_self_modules_blocks_modules_5_modules_9_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_300 = l_self_modules_blocks_modules_5_modules_9_modules_conv_pwl_parameters_weight_ = (None)
        x_302 = torch.nn.functional.batch_norm(
            x_301,
            l_self_modules_blocks_modules_5_modules_9_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_9_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_9_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_9_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_301 = (
            l_self_modules_blocks_modules_5_modules_9_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_9_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_9_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_9_modules_bn3_parameters_bias_
        ) = None
        x_303 = x_302 + x_293
        x_302 = x_293 = None
        x_304 = torch.conv2d(
            x_303,
            l_self_modules_blocks_modules_5_modules_10_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_5_modules_10_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_305 = torch.nn.functional.batch_norm(
            x_304,
            l_self_modules_blocks_modules_5_modules_10_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_10_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_10_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_10_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_304 = (
            l_self_modules_blocks_modules_5_modules_10_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_10_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_10_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_10_modules_bn1_parameters_bias_
        ) = None
        x_306 = torch.nn.functional.silu(x_305, inplace=True)
        x_305 = None
        x_307 = torch.conv2d(
            x_306,
            l_self_modules_blocks_modules_5_modules_10_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1248,
        )
        x_306 = l_self_modules_blocks_modules_5_modules_10_modules_conv_dw_parameters_weight_ = (None)
        x_308 = torch.nn.functional.batch_norm(
            x_307,
            l_self_modules_blocks_modules_5_modules_10_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_10_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_10_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_10_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_307 = (
            l_self_modules_blocks_modules_5_modules_10_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_10_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_10_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_10_modules_bn2_parameters_bias_
        ) = None
        x_309 = torch.nn.functional.silu(x_308, inplace=True)
        x_308 = None
        x_se_100 = x_309.mean((2, 3), keepdim=True)
        x_se_101 = torch.conv2d(
            x_se_100,
            l_self_modules_blocks_modules_5_modules_10_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_10_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_100 = l_self_modules_blocks_modules_5_modules_10_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_10_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_102 = torch.nn.functional.silu(x_se_101, inplace=True)
        x_se_101 = None
        x_se_103 = torch.conv2d(
            x_se_102,
            l_self_modules_blocks_modules_5_modules_10_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_10_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_102 = l_self_modules_blocks_modules_5_modules_10_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_10_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_25 = torch.sigmoid(x_se_103)
        x_se_103 = None
        x_310 = x_309 * sigmoid_25
        x_309 = sigmoid_25 = None
        x_311 = torch.conv2d(
            x_310,
            l_self_modules_blocks_modules_5_modules_10_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_310 = l_self_modules_blocks_modules_5_modules_10_modules_conv_pwl_parameters_weight_ = (None)
        x_312 = torch.nn.functional.batch_norm(
            x_311,
            l_self_modules_blocks_modules_5_modules_10_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_10_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_10_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_10_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_311 = (
            l_self_modules_blocks_modules_5_modules_10_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_10_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_10_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_10_modules_bn3_parameters_bias_
        ) = None
        x_313 = x_312 + x_303
        x_312 = x_303 = None
        x_314 = torch.conv2d(
            x_313,
            l_self_modules_blocks_modules_5_modules_11_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_5_modules_11_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_315 = torch.nn.functional.batch_norm(
            x_314,
            l_self_modules_blocks_modules_5_modules_11_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_11_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_11_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_11_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_314 = (
            l_self_modules_blocks_modules_5_modules_11_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_11_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_11_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_11_modules_bn1_parameters_bias_
        ) = None
        x_316 = torch.nn.functional.silu(x_315, inplace=True)
        x_315 = None
        x_317 = torch.conv2d(
            x_316,
            l_self_modules_blocks_modules_5_modules_11_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1248,
        )
        x_316 = l_self_modules_blocks_modules_5_modules_11_modules_conv_dw_parameters_weight_ = (None)
        x_318 = torch.nn.functional.batch_norm(
            x_317,
            l_self_modules_blocks_modules_5_modules_11_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_11_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_11_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_11_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_317 = (
            l_self_modules_blocks_modules_5_modules_11_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_11_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_11_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_11_modules_bn2_parameters_bias_
        ) = None
        x_319 = torch.nn.functional.silu(x_318, inplace=True)
        x_318 = None
        x_se_104 = x_319.mean((2, 3), keepdim=True)
        x_se_105 = torch.conv2d(
            x_se_104,
            l_self_modules_blocks_modules_5_modules_11_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_11_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_104 = l_self_modules_blocks_modules_5_modules_11_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_11_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_106 = torch.nn.functional.silu(x_se_105, inplace=True)
        x_se_105 = None
        x_se_107 = torch.conv2d(
            x_se_106,
            l_self_modules_blocks_modules_5_modules_11_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_11_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_106 = l_self_modules_blocks_modules_5_modules_11_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_11_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_26 = torch.sigmoid(x_se_107)
        x_se_107 = None
        x_320 = x_319 * sigmoid_26
        x_319 = sigmoid_26 = None
        x_321 = torch.conv2d(
            x_320,
            l_self_modules_blocks_modules_5_modules_11_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_320 = l_self_modules_blocks_modules_5_modules_11_modules_conv_pwl_parameters_weight_ = (None)
        x_322 = torch.nn.functional.batch_norm(
            x_321,
            l_self_modules_blocks_modules_5_modules_11_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_11_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_11_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_11_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_321 = (
            l_self_modules_blocks_modules_5_modules_11_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_11_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_11_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_11_modules_bn3_parameters_bias_
        ) = None
        x_323 = x_322 + x_313
        x_322 = x_313 = None
        x_324 = torch.conv2d(
            x_323,
            l_self_modules_blocks_modules_5_modules_12_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_5_modules_12_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_325 = torch.nn.functional.batch_norm(
            x_324,
            l_self_modules_blocks_modules_5_modules_12_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_12_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_12_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_12_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_324 = (
            l_self_modules_blocks_modules_5_modules_12_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_12_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_12_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_12_modules_bn1_parameters_bias_
        ) = None
        x_326 = torch.nn.functional.silu(x_325, inplace=True)
        x_325 = None
        x_327 = torch.conv2d(
            x_326,
            l_self_modules_blocks_modules_5_modules_12_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1248,
        )
        x_326 = l_self_modules_blocks_modules_5_modules_12_modules_conv_dw_parameters_weight_ = (None)
        x_328 = torch.nn.functional.batch_norm(
            x_327,
            l_self_modules_blocks_modules_5_modules_12_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_12_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_12_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_12_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_327 = (
            l_self_modules_blocks_modules_5_modules_12_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_12_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_12_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_12_modules_bn2_parameters_bias_
        ) = None
        x_329 = torch.nn.functional.silu(x_328, inplace=True)
        x_328 = None
        x_se_108 = x_329.mean((2, 3), keepdim=True)
        x_se_109 = torch.conv2d(
            x_se_108,
            l_self_modules_blocks_modules_5_modules_12_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_12_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_108 = l_self_modules_blocks_modules_5_modules_12_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_12_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_110 = torch.nn.functional.silu(x_se_109, inplace=True)
        x_se_109 = None
        x_se_111 = torch.conv2d(
            x_se_110,
            l_self_modules_blocks_modules_5_modules_12_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_12_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_110 = l_self_modules_blocks_modules_5_modules_12_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_12_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_27 = torch.sigmoid(x_se_111)
        x_se_111 = None
        x_330 = x_329 * sigmoid_27
        x_329 = sigmoid_27 = None
        x_331 = torch.conv2d(
            x_330,
            l_self_modules_blocks_modules_5_modules_12_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_330 = l_self_modules_blocks_modules_5_modules_12_modules_conv_pwl_parameters_weight_ = (None)
        x_332 = torch.nn.functional.batch_norm(
            x_331,
            l_self_modules_blocks_modules_5_modules_12_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_12_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_12_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_12_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_331 = (
            l_self_modules_blocks_modules_5_modules_12_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_12_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_12_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_12_modules_bn3_parameters_bias_
        ) = None
        x_333 = x_332 + x_323
        x_332 = x_323 = None
        x_334 = torch.conv2d(
            x_333,
            l_self_modules_blocks_modules_5_modules_13_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_5_modules_13_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_335 = torch.nn.functional.batch_norm(
            x_334,
            l_self_modules_blocks_modules_5_modules_13_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_13_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_13_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_13_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_334 = (
            l_self_modules_blocks_modules_5_modules_13_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_13_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_13_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_13_modules_bn1_parameters_bias_
        ) = None
        x_336 = torch.nn.functional.silu(x_335, inplace=True)
        x_335 = None
        x_337 = torch.conv2d(
            x_336,
            l_self_modules_blocks_modules_5_modules_13_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1248,
        )
        x_336 = l_self_modules_blocks_modules_5_modules_13_modules_conv_dw_parameters_weight_ = (None)
        x_338 = torch.nn.functional.batch_norm(
            x_337,
            l_self_modules_blocks_modules_5_modules_13_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_13_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_13_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_13_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_337 = (
            l_self_modules_blocks_modules_5_modules_13_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_13_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_13_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_13_modules_bn2_parameters_bias_
        ) = None
        x_339 = torch.nn.functional.silu(x_338, inplace=True)
        x_338 = None
        x_se_112 = x_339.mean((2, 3), keepdim=True)
        x_se_113 = torch.conv2d(
            x_se_112,
            l_self_modules_blocks_modules_5_modules_13_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_13_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_112 = l_self_modules_blocks_modules_5_modules_13_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_13_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_114 = torch.nn.functional.silu(x_se_113, inplace=True)
        x_se_113 = None
        x_se_115 = torch.conv2d(
            x_se_114,
            l_self_modules_blocks_modules_5_modules_13_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_13_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_114 = l_self_modules_blocks_modules_5_modules_13_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_13_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_28 = torch.sigmoid(x_se_115)
        x_se_115 = None
        x_340 = x_339 * sigmoid_28
        x_339 = sigmoid_28 = None
        x_341 = torch.conv2d(
            x_340,
            l_self_modules_blocks_modules_5_modules_13_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_340 = l_self_modules_blocks_modules_5_modules_13_modules_conv_pwl_parameters_weight_ = (None)
        x_342 = torch.nn.functional.batch_norm(
            x_341,
            l_self_modules_blocks_modules_5_modules_13_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_13_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_13_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_13_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_341 = (
            l_self_modules_blocks_modules_5_modules_13_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_13_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_13_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_13_modules_bn3_parameters_bias_
        ) = None
        x_343 = x_342 + x_333
        x_342 = x_333 = None
        x_344 = torch.conv2d(
            x_343,
            l_self_modules_conv_head_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_343 = l_self_modules_conv_head_parameters_weight_ = None
        x_345 = torch.nn.functional.batch_norm(
            x_344,
            l_self_modules_bn2_buffers_running_mean_,
            l_self_modules_bn2_buffers_running_var_,
            l_self_modules_bn2_parameters_weight_,
            l_self_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_344 = (
            l_self_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_bn2_parameters_weight_
        ) = l_self_modules_bn2_parameters_bias_ = None
        x_346 = torch.nn.functional.silu(x_345, inplace=True)
        x_345 = None
        x_347 = torch.nn.functional.adaptive_avg_pool2d(x_346, 1)
        x_346 = None
        x_348 = x_347.flatten(1, -1)
        x_347 = None
        x_349 = torch._C._nn.linear(
            x_348,
            l_self_modules_classifier_parameters_weight_,
            l_self_modules_classifier_parameters_bias_,
        )
        x_348 = (
            l_self_modules_classifier_parameters_weight_
        ) = l_self_modules_classifier_parameters_bias_ = None
        return (x_349,)
