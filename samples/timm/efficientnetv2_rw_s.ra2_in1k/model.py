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
        L_self_modules_blocks_modules_0_modules_0_modules_conv_exp_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_0_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_1_modules_conv_exp_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_1_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_5_modules_14_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_14_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_14_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_14_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_14_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_14_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_14_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_14_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_14_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_14_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_14_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_14_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_14_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_14_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_14_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_14_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_14_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_14_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_14_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_blocks_modules_0_modules_0_modules_conv_exp_parameters_weight_ = L_self_modules_blocks_modules_0_modules_0_modules_conv_exp_parameters_weight_
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
        l_self_modules_blocks_modules_0_modules_0_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_0_modules_0_modules_conv_pwl_parameters_weight_
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
        l_self_modules_blocks_modules_0_modules_1_modules_conv_exp_parameters_weight_ = L_self_modules_blocks_modules_0_modules_1_modules_conv_exp_parameters_weight_
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
        l_self_modules_blocks_modules_0_modules_1_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_0_modules_1_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_0_modules_1_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_0_modules_1_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_0_modules_1_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_0_modules_1_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_0_modules_1_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_1_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_1_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_0_modules_1_modules_bn2_parameters_bias_
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
        l_self_modules_blocks_modules_5_modules_14_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_5_modules_14_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_5_modules_14_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_14_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_14_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_14_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_14_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_14_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_14_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_14_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_14_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_5_modules_14_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_5_modules_14_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_14_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_14_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_14_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_14_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_14_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_14_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_14_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_14_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_5_modules_14_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_5_modules_14_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_5_modules_14_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_5_modules_14_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_5_modules_14_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_5_modules_14_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_5_modules_14_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_5_modules_14_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_5_modules_14_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_5_modules_14_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_14_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_14_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_14_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_14_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_14_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_14_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_14_modules_bn3_parameters_bias_
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
            l_self_modules_blocks_modules_0_modules_0_modules_conv_exp_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_0_modules_0_modules_conv_exp_parameters_weight_ = (
            None
        )
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
        x_6 = torch.conv2d(
            x_5,
            l_self_modules_blocks_modules_0_modules_0_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_5 = l_self_modules_blocks_modules_0_modules_0_modules_conv_pwl_parameters_weight_ = (None)
        x_7 = torch.nn.functional.batch_norm(
            x_6,
            l_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_6 = (
            l_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_bias_
        ) = None
        x_8 = x_7 + x_2
        x_7 = x_2 = None
        x_9 = torch.conv2d(
            x_8,
            l_self_modules_blocks_modules_0_modules_1_modules_conv_exp_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_0_modules_1_modules_conv_exp_parameters_weight_ = (
            None
        )
        x_10 = torch.nn.functional.batch_norm(
            x_9,
            l_self_modules_blocks_modules_0_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_0_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_0_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_9 = (
            l_self_modules_blocks_modules_0_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_0_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_0_modules_1_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_1_modules_bn1_parameters_bias_
        ) = None
        x_11 = torch.nn.functional.silu(x_10, inplace=True)
        x_10 = None
        x_12 = torch.conv2d(
            x_11,
            l_self_modules_blocks_modules_0_modules_1_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_11 = l_self_modules_blocks_modules_0_modules_1_modules_conv_pwl_parameters_weight_ = (None)
        x_13 = torch.nn.functional.batch_norm(
            x_12,
            l_self_modules_blocks_modules_0_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_0_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_0_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_12 = (
            l_self_modules_blocks_modules_0_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_0_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_0_modules_1_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_1_modules_bn2_parameters_bias_
        ) = None
        x_14 = x_13 + x_8
        x_13 = x_8 = None
        x_15 = torch.conv2d(
            x_14,
            l_self_modules_blocks_modules_1_modules_0_modules_conv_exp_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_14 = l_self_modules_blocks_modules_1_modules_0_modules_conv_exp_parameters_weight_ = (None)
        x_16 = torch.nn.functional.batch_norm(
            x_15,
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_15 = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_17 = torch.nn.functional.silu(x_16, inplace=True)
        x_16 = None
        x_18 = torch.conv2d(
            x_17,
            l_self_modules_blocks_modules_1_modules_0_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_17 = l_self_modules_blocks_modules_1_modules_0_modules_conv_pwl_parameters_weight_ = (None)
        x_19 = torch.nn.functional.batch_norm(
            x_18,
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_18 = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_bias_
        ) = None
        x_20 = torch.conv2d(
            x_19,
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
        x_21 = torch.nn.functional.batch_norm(
            x_20,
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_20 = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_bias_
        ) = None
        x_22 = torch.nn.functional.silu(x_21, inplace=True)
        x_21 = None
        x_23 = torch.conv2d(
            x_22,
            l_self_modules_blocks_modules_1_modules_1_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_22 = l_self_modules_blocks_modules_1_modules_1_modules_conv_pwl_parameters_weight_ = (None)
        x_24 = torch.nn.functional.batch_norm(
            x_23,
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_23 = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_bias_
        ) = None
        x_25 = x_24 + x_19
        x_24 = x_19 = None
        x_26 = torch.conv2d(
            x_25,
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
        x_27 = torch.nn.functional.batch_norm(
            x_26,
            l_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_26 = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_bias_
        ) = None
        x_28 = torch.nn.functional.silu(x_27, inplace=True)
        x_27 = None
        x_29 = torch.conv2d(
            x_28,
            l_self_modules_blocks_modules_1_modules_2_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_28 = l_self_modules_blocks_modules_1_modules_2_modules_conv_pwl_parameters_weight_ = (None)
        x_30 = torch.nn.functional.batch_norm(
            x_29,
            l_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_29 = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_bias_
        ) = None
        x_31 = x_30 + x_25
        x_30 = x_25 = None
        x_32 = torch.conv2d(
            x_31,
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
        x_33 = torch.nn.functional.batch_norm(
            x_32,
            l_self_modules_blocks_modules_1_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_32 = (
            l_self_modules_blocks_modules_1_modules_3_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_3_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_3_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_3_modules_bn1_parameters_bias_
        ) = None
        x_34 = torch.nn.functional.silu(x_33, inplace=True)
        x_33 = None
        x_35 = torch.conv2d(
            x_34,
            l_self_modules_blocks_modules_1_modules_3_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_34 = l_self_modules_blocks_modules_1_modules_3_modules_conv_pwl_parameters_weight_ = (None)
        x_36 = torch.nn.functional.batch_norm(
            x_35,
            l_self_modules_blocks_modules_1_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_35 = (
            l_self_modules_blocks_modules_1_modules_3_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_3_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_3_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_3_modules_bn2_parameters_bias_
        ) = None
        x_37 = x_36 + x_31
        x_36 = x_31 = None
        x_38 = torch.conv2d(
            x_37,
            l_self_modules_blocks_modules_2_modules_0_modules_conv_exp_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_37 = l_self_modules_blocks_modules_2_modules_0_modules_conv_exp_parameters_weight_ = (None)
        x_39 = torch.nn.functional.batch_norm(
            x_38,
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_38 = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_40 = torch.nn.functional.silu(x_39, inplace=True)
        x_39 = None
        x_41 = torch.conv2d(
            x_40,
            l_self_modules_blocks_modules_2_modules_0_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_40 = l_self_modules_blocks_modules_2_modules_0_modules_conv_pwl_parameters_weight_ = (None)
        x_42 = torch.nn.functional.batch_norm(
            x_41,
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_41 = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_bias_
        ) = None
        x_43 = torch.conv2d(
            x_42,
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
        x_44 = torch.nn.functional.batch_norm(
            x_43,
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_43 = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_bias_
        ) = None
        x_45 = torch.nn.functional.silu(x_44, inplace=True)
        x_44 = None
        x_46 = torch.conv2d(
            x_45,
            l_self_modules_blocks_modules_2_modules_1_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_45 = l_self_modules_blocks_modules_2_modules_1_modules_conv_pwl_parameters_weight_ = (None)
        x_47 = torch.nn.functional.batch_norm(
            x_46,
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_46 = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_bias_
        ) = None
        x_48 = x_47 + x_42
        x_47 = x_42 = None
        x_49 = torch.conv2d(
            x_48,
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
        x_50 = torch.nn.functional.batch_norm(
            x_49,
            l_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_49 = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_bias_
        ) = None
        x_51 = torch.nn.functional.silu(x_50, inplace=True)
        x_50 = None
        x_52 = torch.conv2d(
            x_51,
            l_self_modules_blocks_modules_2_modules_2_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_51 = l_self_modules_blocks_modules_2_modules_2_modules_conv_pwl_parameters_weight_ = (None)
        x_53 = torch.nn.functional.batch_norm(
            x_52,
            l_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_52 = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_bias_
        ) = None
        x_54 = x_53 + x_48
        x_53 = x_48 = None
        x_55 = torch.conv2d(
            x_54,
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
        x_56 = torch.nn.functional.batch_norm(
            x_55,
            l_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_55 = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_bias_
        ) = None
        x_57 = torch.nn.functional.silu(x_56, inplace=True)
        x_56 = None
        x_58 = torch.conv2d(
            x_57,
            l_self_modules_blocks_modules_2_modules_3_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_57 = l_self_modules_blocks_modules_2_modules_3_modules_conv_pwl_parameters_weight_ = (None)
        x_59 = torch.nn.functional.batch_norm(
            x_58,
            l_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_58 = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_bias_
        ) = None
        x_60 = x_59 + x_54
        x_59 = x_54 = None
        x_61 = torch.conv2d(
            x_60,
            l_self_modules_blocks_modules_3_modules_0_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_60 = (
            l_self_modules_blocks_modules_3_modules_0_modules_conv_pw_parameters_weight_
        ) = None
        x_62 = torch.nn.functional.batch_norm(
            x_61,
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_61 = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_63 = torch.nn.functional.silu(x_62, inplace=True)
        x_62 = None
        x_64 = torch.conv2d(
            x_63,
            l_self_modules_blocks_modules_3_modules_0_modules_conv_dw_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            256,
        )
        x_63 = (
            l_self_modules_blocks_modules_3_modules_0_modules_conv_dw_parameters_weight_
        ) = None
        x_65 = torch.nn.functional.batch_norm(
            x_64,
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_64 = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_bias_
        ) = None
        x_66 = torch.nn.functional.silu(x_65, inplace=True)
        x_65 = None
        x_se = x_66.mean((2, 3), keepdim=True)
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
        x_67 = x_66 * sigmoid
        x_66 = sigmoid = None
        x_68 = torch.conv2d(
            x_67,
            l_self_modules_blocks_modules_3_modules_0_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_67 = l_self_modules_blocks_modules_3_modules_0_modules_conv_pwl_parameters_weight_ = (None)
        x_69 = torch.nn.functional.batch_norm(
            x_68,
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_68 = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_bias_
        ) = None
        x_70 = torch.conv2d(
            x_69,
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
        x_71 = torch.nn.functional.batch_norm(
            x_70,
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_70 = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_bias_
        ) = None
        x_72 = torch.nn.functional.silu(x_71, inplace=True)
        x_71 = None
        x_73 = torch.conv2d(
            x_72,
            l_self_modules_blocks_modules_3_modules_1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            512,
        )
        x_72 = (
            l_self_modules_blocks_modules_3_modules_1_modules_conv_dw_parameters_weight_
        ) = None
        x_74 = torch.nn.functional.batch_norm(
            x_73,
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_73 = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_bias_
        ) = None
        x_75 = torch.nn.functional.silu(x_74, inplace=True)
        x_74 = None
        x_se_4 = x_75.mean((2, 3), keepdim=True)
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
        x_76 = x_75 * sigmoid_1
        x_75 = sigmoid_1 = None
        x_77 = torch.conv2d(
            x_76,
            l_self_modules_blocks_modules_3_modules_1_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_76 = l_self_modules_blocks_modules_3_modules_1_modules_conv_pwl_parameters_weight_ = (None)
        x_78 = torch.nn.functional.batch_norm(
            x_77,
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_77 = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_bias_
        ) = None
        x_79 = x_78 + x_69
        x_78 = x_69 = None
        x_80 = torch.conv2d(
            x_79,
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
        x_81 = torch.nn.functional.batch_norm(
            x_80,
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_80 = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_bias_
        ) = None
        x_82 = torch.nn.functional.silu(x_81, inplace=True)
        x_81 = None
        x_83 = torch.conv2d(
            x_82,
            l_self_modules_blocks_modules_3_modules_2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            512,
        )
        x_82 = (
            l_self_modules_blocks_modules_3_modules_2_modules_conv_dw_parameters_weight_
        ) = None
        x_84 = torch.nn.functional.batch_norm(
            x_83,
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_83 = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_bias_
        ) = None
        x_85 = torch.nn.functional.silu(x_84, inplace=True)
        x_84 = None
        x_se_8 = x_85.mean((2, 3), keepdim=True)
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
        x_86 = x_85 * sigmoid_2
        x_85 = sigmoid_2 = None
        x_87 = torch.conv2d(
            x_86,
            l_self_modules_blocks_modules_3_modules_2_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_86 = l_self_modules_blocks_modules_3_modules_2_modules_conv_pwl_parameters_weight_ = (None)
        x_88 = torch.nn.functional.batch_norm(
            x_87,
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_87 = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_bias_
        ) = None
        x_89 = x_88 + x_79
        x_88 = x_79 = None
        x_90 = torch.conv2d(
            x_89,
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
        x_91 = torch.nn.functional.batch_norm(
            x_90,
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_90 = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_bias_
        ) = None
        x_92 = torch.nn.functional.silu(x_91, inplace=True)
        x_91 = None
        x_93 = torch.conv2d(
            x_92,
            l_self_modules_blocks_modules_3_modules_3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            512,
        )
        x_92 = (
            l_self_modules_blocks_modules_3_modules_3_modules_conv_dw_parameters_weight_
        ) = None
        x_94 = torch.nn.functional.batch_norm(
            x_93,
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_93 = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_bias_
        ) = None
        x_95 = torch.nn.functional.silu(x_94, inplace=True)
        x_94 = None
        x_se_12 = x_95.mean((2, 3), keepdim=True)
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
        x_96 = x_95 * sigmoid_3
        x_95 = sigmoid_3 = None
        x_97 = torch.conv2d(
            x_96,
            l_self_modules_blocks_modules_3_modules_3_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_96 = l_self_modules_blocks_modules_3_modules_3_modules_conv_pwl_parameters_weight_ = (None)
        x_98 = torch.nn.functional.batch_norm(
            x_97,
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_97 = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_bias_
        ) = None
        x_99 = x_98 + x_89
        x_98 = x_89 = None
        x_100 = torch.conv2d(
            x_99,
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
        x_101 = torch.nn.functional.batch_norm(
            x_100,
            l_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_100 = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_bias_
        ) = None
        x_102 = torch.nn.functional.silu(x_101, inplace=True)
        x_101 = None
        x_103 = torch.conv2d(
            x_102,
            l_self_modules_blocks_modules_3_modules_4_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            512,
        )
        x_102 = (
            l_self_modules_blocks_modules_3_modules_4_modules_conv_dw_parameters_weight_
        ) = None
        x_104 = torch.nn.functional.batch_norm(
            x_103,
            l_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_103 = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_bias_
        ) = None
        x_105 = torch.nn.functional.silu(x_104, inplace=True)
        x_104 = None
        x_se_16 = x_105.mean((2, 3), keepdim=True)
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
        x_106 = x_105 * sigmoid_4
        x_105 = sigmoid_4 = None
        x_107 = torch.conv2d(
            x_106,
            l_self_modules_blocks_modules_3_modules_4_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_106 = l_self_modules_blocks_modules_3_modules_4_modules_conv_pwl_parameters_weight_ = (None)
        x_108 = torch.nn.functional.batch_norm(
            x_107,
            l_self_modules_blocks_modules_3_modules_4_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_107 = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn3_parameters_bias_
        ) = None
        x_109 = x_108 + x_99
        x_108 = x_99 = None
        x_110 = torch.conv2d(
            x_109,
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
        x_111 = torch.nn.functional.batch_norm(
            x_110,
            l_self_modules_blocks_modules_3_modules_5_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_5_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_5_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_5_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_110 = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn1_parameters_bias_
        ) = None
        x_112 = torch.nn.functional.silu(x_111, inplace=True)
        x_111 = None
        x_113 = torch.conv2d(
            x_112,
            l_self_modules_blocks_modules_3_modules_5_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            512,
        )
        x_112 = (
            l_self_modules_blocks_modules_3_modules_5_modules_conv_dw_parameters_weight_
        ) = None
        x_114 = torch.nn.functional.batch_norm(
            x_113,
            l_self_modules_blocks_modules_3_modules_5_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_5_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_5_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_5_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_113 = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn2_parameters_bias_
        ) = None
        x_115 = torch.nn.functional.silu(x_114, inplace=True)
        x_114 = None
        x_se_20 = x_115.mean((2, 3), keepdim=True)
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
        x_116 = x_115 * sigmoid_5
        x_115 = sigmoid_5 = None
        x_117 = torch.conv2d(
            x_116,
            l_self_modules_blocks_modules_3_modules_5_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_116 = l_self_modules_blocks_modules_3_modules_5_modules_conv_pwl_parameters_weight_ = (None)
        x_118 = torch.nn.functional.batch_norm(
            x_117,
            l_self_modules_blocks_modules_3_modules_5_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_5_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_5_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_5_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_117 = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn3_parameters_bias_
        ) = None
        x_119 = x_118 + x_109
        x_118 = x_109 = None
        x_120 = torch.conv2d(
            x_119,
            l_self_modules_blocks_modules_4_modules_0_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_119 = (
            l_self_modules_blocks_modules_4_modules_0_modules_conv_pw_parameters_weight_
        ) = None
        x_121 = torch.nn.functional.batch_norm(
            x_120,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_120 = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_122 = torch.nn.functional.silu(x_121, inplace=True)
        x_121 = None
        x_123 = torch.conv2d(
            x_122,
            l_self_modules_blocks_modules_4_modules_0_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        x_122 = (
            l_self_modules_blocks_modules_4_modules_0_modules_conv_dw_parameters_weight_
        ) = None
        x_124 = torch.nn.functional.batch_norm(
            x_123,
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_123 = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_bias_
        ) = None
        x_125 = torch.nn.functional.silu(x_124, inplace=True)
        x_124 = None
        x_se_24 = x_125.mean((2, 3), keepdim=True)
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
        x_126 = x_125 * sigmoid_6
        x_125 = sigmoid_6 = None
        x_127 = torch.conv2d(
            x_126,
            l_self_modules_blocks_modules_4_modules_0_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_126 = l_self_modules_blocks_modules_4_modules_0_modules_conv_pwl_parameters_weight_ = (None)
        x_128 = torch.nn.functional.batch_norm(
            x_127,
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_127 = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_bias_
        ) = None
        x_129 = torch.conv2d(
            x_128,
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
        x_130 = torch.nn.functional.batch_norm(
            x_129,
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_129 = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_bias_
        ) = None
        x_131 = torch.nn.functional.silu(x_130, inplace=True)
        x_130 = None
        x_132 = torch.conv2d(
            x_131,
            l_self_modules_blocks_modules_4_modules_1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            960,
        )
        x_131 = (
            l_self_modules_blocks_modules_4_modules_1_modules_conv_dw_parameters_weight_
        ) = None
        x_133 = torch.nn.functional.batch_norm(
            x_132,
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_132 = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_bias_
        ) = None
        x_134 = torch.nn.functional.silu(x_133, inplace=True)
        x_133 = None
        x_se_28 = x_134.mean((2, 3), keepdim=True)
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
        x_135 = x_134 * sigmoid_7
        x_134 = sigmoid_7 = None
        x_136 = torch.conv2d(
            x_135,
            l_self_modules_blocks_modules_4_modules_1_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_135 = l_self_modules_blocks_modules_4_modules_1_modules_conv_pwl_parameters_weight_ = (None)
        x_137 = torch.nn.functional.batch_norm(
            x_136,
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_136 = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_bias_
        ) = None
        x_138 = x_137 + x_128
        x_137 = x_128 = None
        x_139 = torch.conv2d(
            x_138,
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
        x_140 = torch.nn.functional.batch_norm(
            x_139,
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_139 = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_bias_
        ) = None
        x_141 = torch.nn.functional.silu(x_140, inplace=True)
        x_140 = None
        x_142 = torch.conv2d(
            x_141,
            l_self_modules_blocks_modules_4_modules_2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            960,
        )
        x_141 = (
            l_self_modules_blocks_modules_4_modules_2_modules_conv_dw_parameters_weight_
        ) = None
        x_143 = torch.nn.functional.batch_norm(
            x_142,
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_142 = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_bias_
        ) = None
        x_144 = torch.nn.functional.silu(x_143, inplace=True)
        x_143 = None
        x_se_32 = x_144.mean((2, 3), keepdim=True)
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
        x_145 = x_144 * sigmoid_8
        x_144 = sigmoid_8 = None
        x_146 = torch.conv2d(
            x_145,
            l_self_modules_blocks_modules_4_modules_2_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_145 = l_self_modules_blocks_modules_4_modules_2_modules_conv_pwl_parameters_weight_ = (None)
        x_147 = torch.nn.functional.batch_norm(
            x_146,
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_146 = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_bias_
        ) = None
        x_148 = x_147 + x_138
        x_147 = x_138 = None
        x_149 = torch.conv2d(
            x_148,
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
        x_150 = torch.nn.functional.batch_norm(
            x_149,
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_149 = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_bias_
        ) = None
        x_151 = torch.nn.functional.silu(x_150, inplace=True)
        x_150 = None
        x_152 = torch.conv2d(
            x_151,
            l_self_modules_blocks_modules_4_modules_3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            960,
        )
        x_151 = (
            l_self_modules_blocks_modules_4_modules_3_modules_conv_dw_parameters_weight_
        ) = None
        x_153 = torch.nn.functional.batch_norm(
            x_152,
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_152 = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_bias_
        ) = None
        x_154 = torch.nn.functional.silu(x_153, inplace=True)
        x_153 = None
        x_se_36 = x_154.mean((2, 3), keepdim=True)
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
        x_155 = x_154 * sigmoid_9
        x_154 = sigmoid_9 = None
        x_156 = torch.conv2d(
            x_155,
            l_self_modules_blocks_modules_4_modules_3_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_155 = l_self_modules_blocks_modules_4_modules_3_modules_conv_pwl_parameters_weight_ = (None)
        x_157 = torch.nn.functional.batch_norm(
            x_156,
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_156 = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_bias_
        ) = None
        x_158 = x_157 + x_148
        x_157 = x_148 = None
        x_159 = torch.conv2d(
            x_158,
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
        x_160 = torch.nn.functional.batch_norm(
            x_159,
            l_self_modules_blocks_modules_4_modules_4_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_4_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_4_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_4_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_159 = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn1_parameters_bias_
        ) = None
        x_161 = torch.nn.functional.silu(x_160, inplace=True)
        x_160 = None
        x_162 = torch.conv2d(
            x_161,
            l_self_modules_blocks_modules_4_modules_4_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            960,
        )
        x_161 = (
            l_self_modules_blocks_modules_4_modules_4_modules_conv_dw_parameters_weight_
        ) = None
        x_163 = torch.nn.functional.batch_norm(
            x_162,
            l_self_modules_blocks_modules_4_modules_4_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_4_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_4_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_4_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_162 = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn2_parameters_bias_
        ) = None
        x_164 = torch.nn.functional.silu(x_163, inplace=True)
        x_163 = None
        x_se_40 = x_164.mean((2, 3), keepdim=True)
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
        x_165 = x_164 * sigmoid_10
        x_164 = sigmoid_10 = None
        x_166 = torch.conv2d(
            x_165,
            l_self_modules_blocks_modules_4_modules_4_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_165 = l_self_modules_blocks_modules_4_modules_4_modules_conv_pwl_parameters_weight_ = (None)
        x_167 = torch.nn.functional.batch_norm(
            x_166,
            l_self_modules_blocks_modules_4_modules_4_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_4_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_4_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_4_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_166 = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn3_parameters_bias_
        ) = None
        x_168 = x_167 + x_158
        x_167 = x_158 = None
        x_169 = torch.conv2d(
            x_168,
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
        x_170 = torch.nn.functional.batch_norm(
            x_169,
            l_self_modules_blocks_modules_4_modules_5_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_5_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_5_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_5_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_169 = (
            l_self_modules_blocks_modules_4_modules_5_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_5_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_5_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_5_modules_bn1_parameters_bias_
        ) = None
        x_171 = torch.nn.functional.silu(x_170, inplace=True)
        x_170 = None
        x_172 = torch.conv2d(
            x_171,
            l_self_modules_blocks_modules_4_modules_5_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            960,
        )
        x_171 = (
            l_self_modules_blocks_modules_4_modules_5_modules_conv_dw_parameters_weight_
        ) = None
        x_173 = torch.nn.functional.batch_norm(
            x_172,
            l_self_modules_blocks_modules_4_modules_5_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_5_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_5_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_5_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_172 = (
            l_self_modules_blocks_modules_4_modules_5_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_5_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_5_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_5_modules_bn2_parameters_bias_
        ) = None
        x_174 = torch.nn.functional.silu(x_173, inplace=True)
        x_173 = None
        x_se_44 = x_174.mean((2, 3), keepdim=True)
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
        x_175 = x_174 * sigmoid_11
        x_174 = sigmoid_11 = None
        x_176 = torch.conv2d(
            x_175,
            l_self_modules_blocks_modules_4_modules_5_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_175 = l_self_modules_blocks_modules_4_modules_5_modules_conv_pwl_parameters_weight_ = (None)
        x_177 = torch.nn.functional.batch_norm(
            x_176,
            l_self_modules_blocks_modules_4_modules_5_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_5_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_5_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_5_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_176 = (
            l_self_modules_blocks_modules_4_modules_5_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_5_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_5_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_5_modules_bn3_parameters_bias_
        ) = None
        x_178 = x_177 + x_168
        x_177 = x_168 = None
        x_179 = torch.conv2d(
            x_178,
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
        x_180 = torch.nn.functional.batch_norm(
            x_179,
            l_self_modules_blocks_modules_4_modules_6_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_6_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_6_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_6_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_179 = (
            l_self_modules_blocks_modules_4_modules_6_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_6_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_6_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_6_modules_bn1_parameters_bias_
        ) = None
        x_181 = torch.nn.functional.silu(x_180, inplace=True)
        x_180 = None
        x_182 = torch.conv2d(
            x_181,
            l_self_modules_blocks_modules_4_modules_6_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            960,
        )
        x_181 = (
            l_self_modules_blocks_modules_4_modules_6_modules_conv_dw_parameters_weight_
        ) = None
        x_183 = torch.nn.functional.batch_norm(
            x_182,
            l_self_modules_blocks_modules_4_modules_6_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_6_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_6_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_6_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_182 = (
            l_self_modules_blocks_modules_4_modules_6_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_6_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_6_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_6_modules_bn2_parameters_bias_
        ) = None
        x_184 = torch.nn.functional.silu(x_183, inplace=True)
        x_183 = None
        x_se_48 = x_184.mean((2, 3), keepdim=True)
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
        x_185 = x_184 * sigmoid_12
        x_184 = sigmoid_12 = None
        x_186 = torch.conv2d(
            x_185,
            l_self_modules_blocks_modules_4_modules_6_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_185 = l_self_modules_blocks_modules_4_modules_6_modules_conv_pwl_parameters_weight_ = (None)
        x_187 = torch.nn.functional.batch_norm(
            x_186,
            l_self_modules_blocks_modules_4_modules_6_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_6_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_6_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_6_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_186 = (
            l_self_modules_blocks_modules_4_modules_6_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_6_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_6_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_6_modules_bn3_parameters_bias_
        ) = None
        x_188 = x_187 + x_178
        x_187 = x_178 = None
        x_189 = torch.conv2d(
            x_188,
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
        x_190 = torch.nn.functional.batch_norm(
            x_189,
            l_self_modules_blocks_modules_4_modules_7_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_7_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_7_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_7_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_189 = (
            l_self_modules_blocks_modules_4_modules_7_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_7_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_7_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_7_modules_bn1_parameters_bias_
        ) = None
        x_191 = torch.nn.functional.silu(x_190, inplace=True)
        x_190 = None
        x_192 = torch.conv2d(
            x_191,
            l_self_modules_blocks_modules_4_modules_7_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            960,
        )
        x_191 = (
            l_self_modules_blocks_modules_4_modules_7_modules_conv_dw_parameters_weight_
        ) = None
        x_193 = torch.nn.functional.batch_norm(
            x_192,
            l_self_modules_blocks_modules_4_modules_7_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_7_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_7_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_7_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_192 = (
            l_self_modules_blocks_modules_4_modules_7_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_7_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_7_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_7_modules_bn2_parameters_bias_
        ) = None
        x_194 = torch.nn.functional.silu(x_193, inplace=True)
        x_193 = None
        x_se_52 = x_194.mean((2, 3), keepdim=True)
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
        x_195 = x_194 * sigmoid_13
        x_194 = sigmoid_13 = None
        x_196 = torch.conv2d(
            x_195,
            l_self_modules_blocks_modules_4_modules_7_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_195 = l_self_modules_blocks_modules_4_modules_7_modules_conv_pwl_parameters_weight_ = (None)
        x_197 = torch.nn.functional.batch_norm(
            x_196,
            l_self_modules_blocks_modules_4_modules_7_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_7_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_7_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_7_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_196 = (
            l_self_modules_blocks_modules_4_modules_7_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_7_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_7_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_7_modules_bn3_parameters_bias_
        ) = None
        x_198 = x_197 + x_188
        x_197 = x_188 = None
        x_199 = torch.conv2d(
            x_198,
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
        x_200 = torch.nn.functional.batch_norm(
            x_199,
            l_self_modules_blocks_modules_4_modules_8_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_8_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_8_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_8_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_199 = (
            l_self_modules_blocks_modules_4_modules_8_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_8_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_8_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_8_modules_bn1_parameters_bias_
        ) = None
        x_201 = torch.nn.functional.silu(x_200, inplace=True)
        x_200 = None
        x_202 = torch.conv2d(
            x_201,
            l_self_modules_blocks_modules_4_modules_8_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            960,
        )
        x_201 = (
            l_self_modules_blocks_modules_4_modules_8_modules_conv_dw_parameters_weight_
        ) = None
        x_203 = torch.nn.functional.batch_norm(
            x_202,
            l_self_modules_blocks_modules_4_modules_8_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_8_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_8_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_8_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_202 = (
            l_self_modules_blocks_modules_4_modules_8_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_8_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_8_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_8_modules_bn2_parameters_bias_
        ) = None
        x_204 = torch.nn.functional.silu(x_203, inplace=True)
        x_203 = None
        x_se_56 = x_204.mean((2, 3), keepdim=True)
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
        x_205 = x_204 * sigmoid_14
        x_204 = sigmoid_14 = None
        x_206 = torch.conv2d(
            x_205,
            l_self_modules_blocks_modules_4_modules_8_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_205 = l_self_modules_blocks_modules_4_modules_8_modules_conv_pwl_parameters_weight_ = (None)
        x_207 = torch.nn.functional.batch_norm(
            x_206,
            l_self_modules_blocks_modules_4_modules_8_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_8_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_8_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_8_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_206 = (
            l_self_modules_blocks_modules_4_modules_8_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_8_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_8_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_8_modules_bn3_parameters_bias_
        ) = None
        x_208 = x_207 + x_198
        x_207 = x_198 = None
        x_209 = torch.conv2d(
            x_208,
            l_self_modules_blocks_modules_5_modules_0_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_208 = (
            l_self_modules_blocks_modules_5_modules_0_modules_conv_pw_parameters_weight_
        ) = None
        x_210 = torch.nn.functional.batch_norm(
            x_209,
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_209 = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_211 = torch.nn.functional.silu(x_210, inplace=True)
        x_210 = None
        x_212 = torch.conv2d(
            x_211,
            l_self_modules_blocks_modules_5_modules_0_modules_conv_dw_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            960,
        )
        x_211 = (
            l_self_modules_blocks_modules_5_modules_0_modules_conv_dw_parameters_weight_
        ) = None
        x_213 = torch.nn.functional.batch_norm(
            x_212,
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_212 = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_bias_
        ) = None
        x_214 = torch.nn.functional.silu(x_213, inplace=True)
        x_213 = None
        x_se_60 = x_214.mean((2, 3), keepdim=True)
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
        x_215 = x_214 * sigmoid_15
        x_214 = sigmoid_15 = None
        x_216 = torch.conv2d(
            x_215,
            l_self_modules_blocks_modules_5_modules_0_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_215 = l_self_modules_blocks_modules_5_modules_0_modules_conv_pwl_parameters_weight_ = (None)
        x_217 = torch.nn.functional.batch_norm(
            x_216,
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_216 = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_bias_
        ) = None
        x_218 = torch.conv2d(
            x_217,
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
        x_219 = torch.nn.functional.batch_norm(
            x_218,
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_218 = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_bias_
        ) = None
        x_220 = torch.nn.functional.silu(x_219, inplace=True)
        x_219 = None
        x_221 = torch.conv2d(
            x_220,
            l_self_modules_blocks_modules_5_modules_1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1632,
        )
        x_220 = (
            l_self_modules_blocks_modules_5_modules_1_modules_conv_dw_parameters_weight_
        ) = None
        x_222 = torch.nn.functional.batch_norm(
            x_221,
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_221 = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_bias_
        ) = None
        x_223 = torch.nn.functional.silu(x_222, inplace=True)
        x_222 = None
        x_se_64 = x_223.mean((2, 3), keepdim=True)
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
        x_224 = x_223 * sigmoid_16
        x_223 = sigmoid_16 = None
        x_225 = torch.conv2d(
            x_224,
            l_self_modules_blocks_modules_5_modules_1_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_224 = l_self_modules_blocks_modules_5_modules_1_modules_conv_pwl_parameters_weight_ = (None)
        x_226 = torch.nn.functional.batch_norm(
            x_225,
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_225 = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_bias_
        ) = None
        x_227 = x_226 + x_217
        x_226 = x_217 = None
        x_228 = torch.conv2d(
            x_227,
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
        x_229 = torch.nn.functional.batch_norm(
            x_228,
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_228 = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_bias_
        ) = None
        x_230 = torch.nn.functional.silu(x_229, inplace=True)
        x_229 = None
        x_231 = torch.conv2d(
            x_230,
            l_self_modules_blocks_modules_5_modules_2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1632,
        )
        x_230 = (
            l_self_modules_blocks_modules_5_modules_2_modules_conv_dw_parameters_weight_
        ) = None
        x_232 = torch.nn.functional.batch_norm(
            x_231,
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_231 = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_bias_
        ) = None
        x_233 = torch.nn.functional.silu(x_232, inplace=True)
        x_232 = None
        x_se_68 = x_233.mean((2, 3), keepdim=True)
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
        x_234 = x_233 * sigmoid_17
        x_233 = sigmoid_17 = None
        x_235 = torch.conv2d(
            x_234,
            l_self_modules_blocks_modules_5_modules_2_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_234 = l_self_modules_blocks_modules_5_modules_2_modules_conv_pwl_parameters_weight_ = (None)
        x_236 = torch.nn.functional.batch_norm(
            x_235,
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_235 = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_bias_
        ) = None
        x_237 = x_236 + x_227
        x_236 = x_227 = None
        x_238 = torch.conv2d(
            x_237,
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
        x_239 = torch.nn.functional.batch_norm(
            x_238,
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_238 = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_bias_
        ) = None
        x_240 = torch.nn.functional.silu(x_239, inplace=True)
        x_239 = None
        x_241 = torch.conv2d(
            x_240,
            l_self_modules_blocks_modules_5_modules_3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1632,
        )
        x_240 = (
            l_self_modules_blocks_modules_5_modules_3_modules_conv_dw_parameters_weight_
        ) = None
        x_242 = torch.nn.functional.batch_norm(
            x_241,
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_241 = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_bias_
        ) = None
        x_243 = torch.nn.functional.silu(x_242, inplace=True)
        x_242 = None
        x_se_72 = x_243.mean((2, 3), keepdim=True)
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
        x_244 = x_243 * sigmoid_18
        x_243 = sigmoid_18 = None
        x_245 = torch.conv2d(
            x_244,
            l_self_modules_blocks_modules_5_modules_3_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_244 = l_self_modules_blocks_modules_5_modules_3_modules_conv_pwl_parameters_weight_ = (None)
        x_246 = torch.nn.functional.batch_norm(
            x_245,
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_245 = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_bias_
        ) = None
        x_247 = x_246 + x_237
        x_246 = x_237 = None
        x_248 = torch.conv2d(
            x_247,
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
        x_249 = torch.nn.functional.batch_norm(
            x_248,
            l_self_modules_blocks_modules_5_modules_4_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_4_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_4_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_4_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_248 = (
            l_self_modules_blocks_modules_5_modules_4_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_4_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_4_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_4_modules_bn1_parameters_bias_
        ) = None
        x_250 = torch.nn.functional.silu(x_249, inplace=True)
        x_249 = None
        x_251 = torch.conv2d(
            x_250,
            l_self_modules_blocks_modules_5_modules_4_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1632,
        )
        x_250 = (
            l_self_modules_blocks_modules_5_modules_4_modules_conv_dw_parameters_weight_
        ) = None
        x_252 = torch.nn.functional.batch_norm(
            x_251,
            l_self_modules_blocks_modules_5_modules_4_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_4_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_4_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_4_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_251 = (
            l_self_modules_blocks_modules_5_modules_4_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_4_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_4_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_4_modules_bn2_parameters_bias_
        ) = None
        x_253 = torch.nn.functional.silu(x_252, inplace=True)
        x_252 = None
        x_se_76 = x_253.mean((2, 3), keepdim=True)
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
        x_254 = x_253 * sigmoid_19
        x_253 = sigmoid_19 = None
        x_255 = torch.conv2d(
            x_254,
            l_self_modules_blocks_modules_5_modules_4_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_254 = l_self_modules_blocks_modules_5_modules_4_modules_conv_pwl_parameters_weight_ = (None)
        x_256 = torch.nn.functional.batch_norm(
            x_255,
            l_self_modules_blocks_modules_5_modules_4_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_4_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_4_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_4_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_255 = (
            l_self_modules_blocks_modules_5_modules_4_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_4_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_4_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_4_modules_bn3_parameters_bias_
        ) = None
        x_257 = x_256 + x_247
        x_256 = x_247 = None
        x_258 = torch.conv2d(
            x_257,
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
        x_259 = torch.nn.functional.batch_norm(
            x_258,
            l_self_modules_blocks_modules_5_modules_5_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_5_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_5_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_5_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_258 = (
            l_self_modules_blocks_modules_5_modules_5_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_5_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_5_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_5_modules_bn1_parameters_bias_
        ) = None
        x_260 = torch.nn.functional.silu(x_259, inplace=True)
        x_259 = None
        x_261 = torch.conv2d(
            x_260,
            l_self_modules_blocks_modules_5_modules_5_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1632,
        )
        x_260 = (
            l_self_modules_blocks_modules_5_modules_5_modules_conv_dw_parameters_weight_
        ) = None
        x_262 = torch.nn.functional.batch_norm(
            x_261,
            l_self_modules_blocks_modules_5_modules_5_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_5_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_5_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_5_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_261 = (
            l_self_modules_blocks_modules_5_modules_5_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_5_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_5_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_5_modules_bn2_parameters_bias_
        ) = None
        x_263 = torch.nn.functional.silu(x_262, inplace=True)
        x_262 = None
        x_se_80 = x_263.mean((2, 3), keepdim=True)
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
        x_264 = x_263 * sigmoid_20
        x_263 = sigmoid_20 = None
        x_265 = torch.conv2d(
            x_264,
            l_self_modules_blocks_modules_5_modules_5_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_264 = l_self_modules_blocks_modules_5_modules_5_modules_conv_pwl_parameters_weight_ = (None)
        x_266 = torch.nn.functional.batch_norm(
            x_265,
            l_self_modules_blocks_modules_5_modules_5_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_5_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_5_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_5_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_265 = (
            l_self_modules_blocks_modules_5_modules_5_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_5_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_5_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_5_modules_bn3_parameters_bias_
        ) = None
        x_267 = x_266 + x_257
        x_266 = x_257 = None
        x_268 = torch.conv2d(
            x_267,
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
        x_269 = torch.nn.functional.batch_norm(
            x_268,
            l_self_modules_blocks_modules_5_modules_6_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_6_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_6_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_6_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_268 = (
            l_self_modules_blocks_modules_5_modules_6_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_6_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_6_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_6_modules_bn1_parameters_bias_
        ) = None
        x_270 = torch.nn.functional.silu(x_269, inplace=True)
        x_269 = None
        x_271 = torch.conv2d(
            x_270,
            l_self_modules_blocks_modules_5_modules_6_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1632,
        )
        x_270 = (
            l_self_modules_blocks_modules_5_modules_6_modules_conv_dw_parameters_weight_
        ) = None
        x_272 = torch.nn.functional.batch_norm(
            x_271,
            l_self_modules_blocks_modules_5_modules_6_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_6_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_6_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_6_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_271 = (
            l_self_modules_blocks_modules_5_modules_6_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_6_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_6_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_6_modules_bn2_parameters_bias_
        ) = None
        x_273 = torch.nn.functional.silu(x_272, inplace=True)
        x_272 = None
        x_se_84 = x_273.mean((2, 3), keepdim=True)
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
        x_274 = x_273 * sigmoid_21
        x_273 = sigmoid_21 = None
        x_275 = torch.conv2d(
            x_274,
            l_self_modules_blocks_modules_5_modules_6_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_274 = l_self_modules_blocks_modules_5_modules_6_modules_conv_pwl_parameters_weight_ = (None)
        x_276 = torch.nn.functional.batch_norm(
            x_275,
            l_self_modules_blocks_modules_5_modules_6_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_6_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_6_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_6_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_275 = (
            l_self_modules_blocks_modules_5_modules_6_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_6_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_6_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_6_modules_bn3_parameters_bias_
        ) = None
        x_277 = x_276 + x_267
        x_276 = x_267 = None
        x_278 = torch.conv2d(
            x_277,
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
        x_279 = torch.nn.functional.batch_norm(
            x_278,
            l_self_modules_blocks_modules_5_modules_7_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_7_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_7_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_7_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_278 = (
            l_self_modules_blocks_modules_5_modules_7_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_7_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_7_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_7_modules_bn1_parameters_bias_
        ) = None
        x_280 = torch.nn.functional.silu(x_279, inplace=True)
        x_279 = None
        x_281 = torch.conv2d(
            x_280,
            l_self_modules_blocks_modules_5_modules_7_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1632,
        )
        x_280 = (
            l_self_modules_blocks_modules_5_modules_7_modules_conv_dw_parameters_weight_
        ) = None
        x_282 = torch.nn.functional.batch_norm(
            x_281,
            l_self_modules_blocks_modules_5_modules_7_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_7_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_7_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_7_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_281 = (
            l_self_modules_blocks_modules_5_modules_7_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_7_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_7_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_7_modules_bn2_parameters_bias_
        ) = None
        x_283 = torch.nn.functional.silu(x_282, inplace=True)
        x_282 = None
        x_se_88 = x_283.mean((2, 3), keepdim=True)
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
        x_284 = x_283 * sigmoid_22
        x_283 = sigmoid_22 = None
        x_285 = torch.conv2d(
            x_284,
            l_self_modules_blocks_modules_5_modules_7_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_284 = l_self_modules_blocks_modules_5_modules_7_modules_conv_pwl_parameters_weight_ = (None)
        x_286 = torch.nn.functional.batch_norm(
            x_285,
            l_self_modules_blocks_modules_5_modules_7_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_7_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_7_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_7_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_285 = (
            l_self_modules_blocks_modules_5_modules_7_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_7_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_7_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_7_modules_bn3_parameters_bias_
        ) = None
        x_287 = x_286 + x_277
        x_286 = x_277 = None
        x_288 = torch.conv2d(
            x_287,
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
        x_289 = torch.nn.functional.batch_norm(
            x_288,
            l_self_modules_blocks_modules_5_modules_8_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_8_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_8_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_8_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_288 = (
            l_self_modules_blocks_modules_5_modules_8_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_8_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_8_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_8_modules_bn1_parameters_bias_
        ) = None
        x_290 = torch.nn.functional.silu(x_289, inplace=True)
        x_289 = None
        x_291 = torch.conv2d(
            x_290,
            l_self_modules_blocks_modules_5_modules_8_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1632,
        )
        x_290 = (
            l_self_modules_blocks_modules_5_modules_8_modules_conv_dw_parameters_weight_
        ) = None
        x_292 = torch.nn.functional.batch_norm(
            x_291,
            l_self_modules_blocks_modules_5_modules_8_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_8_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_8_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_8_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_291 = (
            l_self_modules_blocks_modules_5_modules_8_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_8_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_8_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_8_modules_bn2_parameters_bias_
        ) = None
        x_293 = torch.nn.functional.silu(x_292, inplace=True)
        x_292 = None
        x_se_92 = x_293.mean((2, 3), keepdim=True)
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
        x_294 = x_293 * sigmoid_23
        x_293 = sigmoid_23 = None
        x_295 = torch.conv2d(
            x_294,
            l_self_modules_blocks_modules_5_modules_8_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_294 = l_self_modules_blocks_modules_5_modules_8_modules_conv_pwl_parameters_weight_ = (None)
        x_296 = torch.nn.functional.batch_norm(
            x_295,
            l_self_modules_blocks_modules_5_modules_8_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_8_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_8_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_8_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_295 = (
            l_self_modules_blocks_modules_5_modules_8_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_8_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_8_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_8_modules_bn3_parameters_bias_
        ) = None
        x_297 = x_296 + x_287
        x_296 = x_287 = None
        x_298 = torch.conv2d(
            x_297,
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
        x_299 = torch.nn.functional.batch_norm(
            x_298,
            l_self_modules_blocks_modules_5_modules_9_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_9_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_9_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_9_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_298 = (
            l_self_modules_blocks_modules_5_modules_9_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_9_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_9_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_9_modules_bn1_parameters_bias_
        ) = None
        x_300 = torch.nn.functional.silu(x_299, inplace=True)
        x_299 = None
        x_301 = torch.conv2d(
            x_300,
            l_self_modules_blocks_modules_5_modules_9_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1632,
        )
        x_300 = (
            l_self_modules_blocks_modules_5_modules_9_modules_conv_dw_parameters_weight_
        ) = None
        x_302 = torch.nn.functional.batch_norm(
            x_301,
            l_self_modules_blocks_modules_5_modules_9_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_9_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_9_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_9_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_301 = (
            l_self_modules_blocks_modules_5_modules_9_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_9_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_9_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_9_modules_bn2_parameters_bias_
        ) = None
        x_303 = torch.nn.functional.silu(x_302, inplace=True)
        x_302 = None
        x_se_96 = x_303.mean((2, 3), keepdim=True)
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
        x_304 = x_303 * sigmoid_24
        x_303 = sigmoid_24 = None
        x_305 = torch.conv2d(
            x_304,
            l_self_modules_blocks_modules_5_modules_9_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_304 = l_self_modules_blocks_modules_5_modules_9_modules_conv_pwl_parameters_weight_ = (None)
        x_306 = torch.nn.functional.batch_norm(
            x_305,
            l_self_modules_blocks_modules_5_modules_9_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_9_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_9_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_9_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_305 = (
            l_self_modules_blocks_modules_5_modules_9_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_9_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_9_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_9_modules_bn3_parameters_bias_
        ) = None
        x_307 = x_306 + x_297
        x_306 = x_297 = None
        x_308 = torch.conv2d(
            x_307,
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
        x_309 = torch.nn.functional.batch_norm(
            x_308,
            l_self_modules_blocks_modules_5_modules_10_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_10_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_10_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_10_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_308 = (
            l_self_modules_blocks_modules_5_modules_10_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_10_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_10_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_10_modules_bn1_parameters_bias_
        ) = None
        x_310 = torch.nn.functional.silu(x_309, inplace=True)
        x_309 = None
        x_311 = torch.conv2d(
            x_310,
            l_self_modules_blocks_modules_5_modules_10_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1632,
        )
        x_310 = l_self_modules_blocks_modules_5_modules_10_modules_conv_dw_parameters_weight_ = (None)
        x_312 = torch.nn.functional.batch_norm(
            x_311,
            l_self_modules_blocks_modules_5_modules_10_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_10_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_10_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_10_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_311 = (
            l_self_modules_blocks_modules_5_modules_10_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_10_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_10_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_10_modules_bn2_parameters_bias_
        ) = None
        x_313 = torch.nn.functional.silu(x_312, inplace=True)
        x_312 = None
        x_se_100 = x_313.mean((2, 3), keepdim=True)
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
        x_314 = x_313 * sigmoid_25
        x_313 = sigmoid_25 = None
        x_315 = torch.conv2d(
            x_314,
            l_self_modules_blocks_modules_5_modules_10_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_314 = l_self_modules_blocks_modules_5_modules_10_modules_conv_pwl_parameters_weight_ = (None)
        x_316 = torch.nn.functional.batch_norm(
            x_315,
            l_self_modules_blocks_modules_5_modules_10_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_10_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_10_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_10_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_315 = (
            l_self_modules_blocks_modules_5_modules_10_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_10_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_10_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_10_modules_bn3_parameters_bias_
        ) = None
        x_317 = x_316 + x_307
        x_316 = x_307 = None
        x_318 = torch.conv2d(
            x_317,
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
        x_319 = torch.nn.functional.batch_norm(
            x_318,
            l_self_modules_blocks_modules_5_modules_11_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_11_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_11_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_11_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_318 = (
            l_self_modules_blocks_modules_5_modules_11_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_11_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_11_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_11_modules_bn1_parameters_bias_
        ) = None
        x_320 = torch.nn.functional.silu(x_319, inplace=True)
        x_319 = None
        x_321 = torch.conv2d(
            x_320,
            l_self_modules_blocks_modules_5_modules_11_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1632,
        )
        x_320 = l_self_modules_blocks_modules_5_modules_11_modules_conv_dw_parameters_weight_ = (None)
        x_322 = torch.nn.functional.batch_norm(
            x_321,
            l_self_modules_blocks_modules_5_modules_11_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_11_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_11_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_11_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_321 = (
            l_self_modules_blocks_modules_5_modules_11_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_11_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_11_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_11_modules_bn2_parameters_bias_
        ) = None
        x_323 = torch.nn.functional.silu(x_322, inplace=True)
        x_322 = None
        x_se_104 = x_323.mean((2, 3), keepdim=True)
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
        x_324 = x_323 * sigmoid_26
        x_323 = sigmoid_26 = None
        x_325 = torch.conv2d(
            x_324,
            l_self_modules_blocks_modules_5_modules_11_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_324 = l_self_modules_blocks_modules_5_modules_11_modules_conv_pwl_parameters_weight_ = (None)
        x_326 = torch.nn.functional.batch_norm(
            x_325,
            l_self_modules_blocks_modules_5_modules_11_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_11_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_11_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_11_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_325 = (
            l_self_modules_blocks_modules_5_modules_11_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_11_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_11_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_11_modules_bn3_parameters_bias_
        ) = None
        x_327 = x_326 + x_317
        x_326 = x_317 = None
        x_328 = torch.conv2d(
            x_327,
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
        x_329 = torch.nn.functional.batch_norm(
            x_328,
            l_self_modules_blocks_modules_5_modules_12_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_12_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_12_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_12_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_328 = (
            l_self_modules_blocks_modules_5_modules_12_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_12_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_12_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_12_modules_bn1_parameters_bias_
        ) = None
        x_330 = torch.nn.functional.silu(x_329, inplace=True)
        x_329 = None
        x_331 = torch.conv2d(
            x_330,
            l_self_modules_blocks_modules_5_modules_12_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1632,
        )
        x_330 = l_self_modules_blocks_modules_5_modules_12_modules_conv_dw_parameters_weight_ = (None)
        x_332 = torch.nn.functional.batch_norm(
            x_331,
            l_self_modules_blocks_modules_5_modules_12_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_12_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_12_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_12_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_331 = (
            l_self_modules_blocks_modules_5_modules_12_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_12_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_12_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_12_modules_bn2_parameters_bias_
        ) = None
        x_333 = torch.nn.functional.silu(x_332, inplace=True)
        x_332 = None
        x_se_108 = x_333.mean((2, 3), keepdim=True)
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
        x_334 = x_333 * sigmoid_27
        x_333 = sigmoid_27 = None
        x_335 = torch.conv2d(
            x_334,
            l_self_modules_blocks_modules_5_modules_12_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_334 = l_self_modules_blocks_modules_5_modules_12_modules_conv_pwl_parameters_weight_ = (None)
        x_336 = torch.nn.functional.batch_norm(
            x_335,
            l_self_modules_blocks_modules_5_modules_12_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_12_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_12_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_12_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_335 = (
            l_self_modules_blocks_modules_5_modules_12_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_12_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_12_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_12_modules_bn3_parameters_bias_
        ) = None
        x_337 = x_336 + x_327
        x_336 = x_327 = None
        x_338 = torch.conv2d(
            x_337,
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
        x_339 = torch.nn.functional.batch_norm(
            x_338,
            l_self_modules_blocks_modules_5_modules_13_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_13_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_13_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_13_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_338 = (
            l_self_modules_blocks_modules_5_modules_13_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_13_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_13_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_13_modules_bn1_parameters_bias_
        ) = None
        x_340 = torch.nn.functional.silu(x_339, inplace=True)
        x_339 = None
        x_341 = torch.conv2d(
            x_340,
            l_self_modules_blocks_modules_5_modules_13_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1632,
        )
        x_340 = l_self_modules_blocks_modules_5_modules_13_modules_conv_dw_parameters_weight_ = (None)
        x_342 = torch.nn.functional.batch_norm(
            x_341,
            l_self_modules_blocks_modules_5_modules_13_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_13_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_13_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_13_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_341 = (
            l_self_modules_blocks_modules_5_modules_13_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_13_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_13_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_13_modules_bn2_parameters_bias_
        ) = None
        x_343 = torch.nn.functional.silu(x_342, inplace=True)
        x_342 = None
        x_se_112 = x_343.mean((2, 3), keepdim=True)
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
        x_344 = x_343 * sigmoid_28
        x_343 = sigmoid_28 = None
        x_345 = torch.conv2d(
            x_344,
            l_self_modules_blocks_modules_5_modules_13_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_344 = l_self_modules_blocks_modules_5_modules_13_modules_conv_pwl_parameters_weight_ = (None)
        x_346 = torch.nn.functional.batch_norm(
            x_345,
            l_self_modules_blocks_modules_5_modules_13_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_13_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_13_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_13_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_345 = (
            l_self_modules_blocks_modules_5_modules_13_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_13_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_13_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_13_modules_bn3_parameters_bias_
        ) = None
        x_347 = x_346 + x_337
        x_346 = x_337 = None
        x_348 = torch.conv2d(
            x_347,
            l_self_modules_blocks_modules_5_modules_14_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_5_modules_14_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_349 = torch.nn.functional.batch_norm(
            x_348,
            l_self_modules_blocks_modules_5_modules_14_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_14_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_14_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_14_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_348 = (
            l_self_modules_blocks_modules_5_modules_14_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_14_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_14_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_14_modules_bn1_parameters_bias_
        ) = None
        x_350 = torch.nn.functional.silu(x_349, inplace=True)
        x_349 = None
        x_351 = torch.conv2d(
            x_350,
            l_self_modules_blocks_modules_5_modules_14_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1632,
        )
        x_350 = l_self_modules_blocks_modules_5_modules_14_modules_conv_dw_parameters_weight_ = (None)
        x_352 = torch.nn.functional.batch_norm(
            x_351,
            l_self_modules_blocks_modules_5_modules_14_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_14_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_14_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_14_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_351 = (
            l_self_modules_blocks_modules_5_modules_14_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_14_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_14_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_14_modules_bn2_parameters_bias_
        ) = None
        x_353 = torch.nn.functional.silu(x_352, inplace=True)
        x_352 = None
        x_se_116 = x_353.mean((2, 3), keepdim=True)
        x_se_117 = torch.conv2d(
            x_se_116,
            l_self_modules_blocks_modules_5_modules_14_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_14_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_116 = l_self_modules_blocks_modules_5_modules_14_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_14_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_118 = torch.nn.functional.silu(x_se_117, inplace=True)
        x_se_117 = None
        x_se_119 = torch.conv2d(
            x_se_118,
            l_self_modules_blocks_modules_5_modules_14_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_14_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_118 = l_self_modules_blocks_modules_5_modules_14_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_14_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_29 = torch.sigmoid(x_se_119)
        x_se_119 = None
        x_354 = x_353 * sigmoid_29
        x_353 = sigmoid_29 = None
        x_355 = torch.conv2d(
            x_354,
            l_self_modules_blocks_modules_5_modules_14_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_354 = l_self_modules_blocks_modules_5_modules_14_modules_conv_pwl_parameters_weight_ = (None)
        x_356 = torch.nn.functional.batch_norm(
            x_355,
            l_self_modules_blocks_modules_5_modules_14_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_14_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_14_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_14_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_355 = (
            l_self_modules_blocks_modules_5_modules_14_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_14_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_14_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_14_modules_bn3_parameters_bias_
        ) = None
        x_357 = x_356 + x_347
        x_356 = x_347 = None
        x_358 = torch.conv2d(
            x_357,
            l_self_modules_conv_head_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_357 = l_self_modules_conv_head_parameters_weight_ = None
        x_359 = torch.nn.functional.batch_norm(
            x_358,
            l_self_modules_bn2_buffers_running_mean_,
            l_self_modules_bn2_buffers_running_var_,
            l_self_modules_bn2_parameters_weight_,
            l_self_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_358 = (
            l_self_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_bn2_parameters_weight_
        ) = l_self_modules_bn2_parameters_bias_ = None
        x_360 = torch.nn.functional.silu(x_359, inplace=True)
        x_359 = None
        x_361 = torch.nn.functional.adaptive_avg_pool2d(x_360, 1)
        x_360 = None
        x_362 = x_361.flatten(1, -1)
        x_361 = None
        x_363 = torch._C._nn.linear(
            x_362,
            l_self_modules_classifier_parameters_weight_,
            l_self_modules_classifier_parameters_bias_,
        )
        x_362 = (
            l_self_modules_classifier_parameters_weight_
        ) = l_self_modules_classifier_parameters_bias_ = None
        return (x_363,)
