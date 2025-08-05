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
        L_self_modules_blocks_modules_0_modules_2_modules_conv_exp_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_2_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_1_modules_4_modules_conv_exp_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_4_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_4_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_4_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_4_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_4_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_4_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_4_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_4_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_4_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_2_modules_4_modules_conv_exp_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_4_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_4_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_4_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_4_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_4_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_4_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_4_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_4_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_4_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_3_modules_6_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_6_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_6_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_6_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_6_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_6_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_6_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_6_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_6_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_6_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_6_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_6_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_6_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_6_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_6_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_6_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_6_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_6_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_6_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_7_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_7_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_7_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_7_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_7_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_7_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_7_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_7_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_7_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_7_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_7_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_7_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_7_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_7_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_7_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_7_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_7_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_7_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_7_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_4_modules_9_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_9_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_9_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_9_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_9_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_9_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_9_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_9_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_9_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_9_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_9_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_9_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_9_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_9_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_9_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_9_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_9_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_9_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_9_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_10_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_10_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_10_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_10_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_10_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_10_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_10_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_10_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_10_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_10_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_10_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_10_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_10_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_10_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_10_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_10_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_10_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_10_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_10_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_11_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_11_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_11_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_11_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_11_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_11_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_11_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_11_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_11_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_11_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_11_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_11_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_11_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_11_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_11_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_11_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_11_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_11_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_11_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_12_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_12_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_12_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_12_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_12_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_12_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_12_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_12_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_12_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_12_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_12_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_12_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_12_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_12_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_12_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_12_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_12_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_12_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_12_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_13_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_13_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_13_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_13_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_13_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_13_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_13_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_13_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_13_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_13_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_13_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_13_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_13_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_13_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_13_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_13_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_13_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_13_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_13_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_14_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_14_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_14_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_14_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_14_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_14_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_14_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_14_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_14_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_14_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_14_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_14_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_14_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_14_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_14_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_14_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_14_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_14_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_14_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_5_modules_15_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_15_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_15_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_15_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_15_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_15_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_15_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_15_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_15_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_15_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_15_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_15_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_15_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_15_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_15_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_15_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_15_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_15_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_15_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_16_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_16_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_16_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_16_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_16_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_16_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_16_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_16_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_16_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_16_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_16_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_16_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_16_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_16_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_16_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_16_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_16_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_16_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_16_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_17_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_17_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_17_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_17_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_17_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_17_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_17_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_17_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_17_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_17_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_17_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_17_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_17_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_17_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_17_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_17_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_17_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_17_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_17_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_18_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_18_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_18_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_18_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_18_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_18_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_18_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_18_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_18_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_18_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_18_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_18_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_18_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_18_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_18_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_18_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_18_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_18_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_18_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_19_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_19_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_19_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_19_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_19_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_19_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_19_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_19_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_19_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_19_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_19_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_19_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_19_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_19_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_19_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_19_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_19_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_19_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_19_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_20_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_20_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_20_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_20_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_20_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_20_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_20_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_20_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_20_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_20_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_20_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_20_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_20_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_20_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_20_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_20_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_20_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_20_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_20_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_21_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_21_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_21_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_21_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_21_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_21_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_21_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_21_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_21_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_21_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_21_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_21_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_21_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_21_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_21_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_21_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_21_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_21_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_21_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_22_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_22_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_22_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_22_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_22_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_22_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_22_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_22_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_22_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_22_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_22_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_22_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_22_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_22_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_22_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_22_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_22_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_22_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_22_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_23_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_23_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_23_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_23_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_23_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_23_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_23_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_23_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_23_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_23_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_23_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_23_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_23_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_23_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_23_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_23_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_23_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_23_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_23_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_blocks_modules_0_modules_2_modules_conv_exp_parameters_weight_ = L_self_modules_blocks_modules_0_modules_2_modules_conv_exp_parameters_weight_
        l_self_modules_blocks_modules_0_modules_2_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_0_modules_2_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_0_modules_2_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_0_modules_2_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_0_modules_2_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_2_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_2_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_0_modules_2_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_0_modules_2_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_0_modules_2_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_0_modules_2_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_0_modules_2_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_0_modules_2_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_0_modules_2_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_0_modules_2_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_2_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_2_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_0_modules_2_modules_bn2_parameters_bias_
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
        l_self_modules_blocks_modules_1_modules_4_modules_conv_exp_parameters_weight_ = L_self_modules_blocks_modules_1_modules_4_modules_conv_exp_parameters_weight_
        l_self_modules_blocks_modules_1_modules_4_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_1_modules_4_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_1_modules_4_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_1_modules_4_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_1_modules_4_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_4_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_4_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_4_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_4_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_1_modules_4_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_1_modules_4_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_1_modules_4_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_1_modules_4_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_1_modules_4_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_1_modules_4_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_4_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_4_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_4_modules_bn2_parameters_bias_
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
        l_self_modules_blocks_modules_2_modules_4_modules_conv_exp_parameters_weight_ = L_self_modules_blocks_modules_2_modules_4_modules_conv_exp_parameters_weight_
        l_self_modules_blocks_modules_2_modules_4_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_2_modules_4_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_2_modules_4_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_2_modules_4_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_2_modules_4_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_4_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_4_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_4_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_4_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_2_modules_4_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_2_modules_4_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_2_modules_4_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_2_modules_4_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_2_modules_4_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_2_modules_4_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_4_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_4_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_4_modules_bn2_parameters_bias_
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
        l_self_modules_blocks_modules_3_modules_6_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_6_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_6_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_3_modules_6_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_3_modules_6_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_3_modules_6_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_3_modules_6_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_6_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_6_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_6_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_6_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_6_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_6_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_3_modules_6_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_3_modules_6_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_3_modules_6_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_3_modules_6_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_6_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_6_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_6_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_6_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_3_modules_6_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_3_modules_6_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_3_modules_6_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_3_modules_6_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_3_modules_6_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_3_modules_6_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_3_modules_6_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_3_modules_6_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_3_modules_6_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_3_modules_6_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_3_modules_6_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_3_modules_6_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_3_modules_6_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_3_modules_6_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_6_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_6_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_6_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_7_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_7_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_7_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_3_modules_7_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_3_modules_7_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_3_modules_7_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_3_modules_7_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_7_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_7_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_7_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_7_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_7_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_7_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_3_modules_7_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_3_modules_7_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_3_modules_7_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_3_modules_7_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_7_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_7_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_7_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_7_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_3_modules_7_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_3_modules_7_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_3_modules_7_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_3_modules_7_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_3_modules_7_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_3_modules_7_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_3_modules_7_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_3_modules_7_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_3_modules_7_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_3_modules_7_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_3_modules_7_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_3_modules_7_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_3_modules_7_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_3_modules_7_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_7_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_7_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_7_modules_bn3_parameters_bias_
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
        l_self_modules_blocks_modules_4_modules_9_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_9_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_9_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_9_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_9_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_9_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_9_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_9_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_9_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_9_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_9_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_9_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_9_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_9_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_9_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_9_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_9_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_9_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_9_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_9_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_9_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_4_modules_9_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_4_modules_9_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_4_modules_9_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_4_modules_9_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_4_modules_9_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_4_modules_9_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_4_modules_9_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_4_modules_9_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_4_modules_9_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_4_modules_9_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_9_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_9_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_9_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_9_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_9_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_9_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_9_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_10_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_4_modules_10_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_4_modules_10_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_10_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_10_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_10_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_10_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_10_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_10_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_10_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_10_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_4_modules_10_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_4_modules_10_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_10_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_10_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_10_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_10_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_10_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_10_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_10_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_10_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_4_modules_10_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_4_modules_10_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_4_modules_10_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_4_modules_10_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_4_modules_10_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_4_modules_10_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_4_modules_10_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_4_modules_10_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_4_modules_10_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_4_modules_10_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_10_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_10_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_10_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_10_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_10_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_10_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_10_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_11_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_4_modules_11_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_4_modules_11_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_11_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_11_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_11_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_11_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_11_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_11_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_11_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_11_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_4_modules_11_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_4_modules_11_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_11_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_11_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_11_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_11_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_11_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_11_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_11_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_11_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_4_modules_11_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_4_modules_11_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_4_modules_11_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_4_modules_11_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_4_modules_11_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_4_modules_11_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_4_modules_11_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_4_modules_11_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_4_modules_11_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_4_modules_11_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_11_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_11_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_11_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_11_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_11_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_11_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_11_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_12_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_4_modules_12_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_4_modules_12_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_12_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_12_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_12_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_12_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_12_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_12_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_12_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_12_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_4_modules_12_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_4_modules_12_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_12_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_12_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_12_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_12_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_12_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_12_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_12_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_12_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_4_modules_12_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_4_modules_12_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_4_modules_12_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_4_modules_12_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_4_modules_12_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_4_modules_12_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_4_modules_12_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_4_modules_12_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_4_modules_12_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_4_modules_12_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_12_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_12_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_12_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_12_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_12_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_12_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_12_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_13_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_4_modules_13_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_4_modules_13_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_13_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_13_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_13_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_13_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_13_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_13_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_13_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_13_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_4_modules_13_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_4_modules_13_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_13_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_13_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_13_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_13_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_13_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_13_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_13_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_13_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_4_modules_13_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_4_modules_13_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_4_modules_13_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_4_modules_13_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_4_modules_13_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_4_modules_13_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_4_modules_13_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_4_modules_13_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_4_modules_13_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_4_modules_13_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_13_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_13_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_13_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_13_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_13_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_13_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_13_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_14_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_4_modules_14_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_4_modules_14_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_14_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_14_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_14_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_14_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_14_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_14_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_14_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_14_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_4_modules_14_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_4_modules_14_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_14_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_14_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_14_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_14_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_14_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_14_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_14_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_14_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_4_modules_14_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_4_modules_14_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_4_modules_14_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_4_modules_14_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_4_modules_14_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_4_modules_14_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_4_modules_14_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_4_modules_14_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_4_modules_14_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_4_modules_14_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_14_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_14_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_14_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_14_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_14_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_14_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_14_modules_bn3_parameters_bias_
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
        l_self_modules_blocks_modules_5_modules_15_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_5_modules_15_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_5_modules_15_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_15_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_15_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_15_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_15_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_15_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_15_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_15_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_15_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_5_modules_15_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_5_modules_15_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_15_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_15_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_15_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_15_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_15_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_15_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_15_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_15_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_5_modules_15_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_5_modules_15_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_5_modules_15_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_5_modules_15_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_5_modules_15_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_5_modules_15_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_5_modules_15_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_5_modules_15_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_5_modules_15_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_5_modules_15_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_15_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_15_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_15_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_15_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_15_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_15_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_15_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_16_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_5_modules_16_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_5_modules_16_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_16_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_16_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_16_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_16_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_16_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_16_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_16_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_16_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_5_modules_16_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_5_modules_16_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_16_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_16_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_16_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_16_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_16_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_16_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_16_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_16_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_5_modules_16_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_5_modules_16_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_5_modules_16_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_5_modules_16_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_5_modules_16_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_5_modules_16_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_5_modules_16_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_5_modules_16_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_5_modules_16_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_5_modules_16_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_16_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_16_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_16_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_16_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_16_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_16_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_16_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_17_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_5_modules_17_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_5_modules_17_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_17_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_17_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_17_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_17_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_17_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_17_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_17_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_17_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_5_modules_17_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_5_modules_17_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_17_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_17_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_17_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_17_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_17_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_17_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_17_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_17_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_5_modules_17_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_5_modules_17_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_5_modules_17_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_5_modules_17_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_5_modules_17_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_5_modules_17_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_5_modules_17_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_5_modules_17_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_5_modules_17_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_5_modules_17_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_17_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_17_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_17_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_17_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_17_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_17_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_17_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_18_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_5_modules_18_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_5_modules_18_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_18_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_18_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_18_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_18_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_18_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_18_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_18_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_18_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_5_modules_18_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_5_modules_18_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_18_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_18_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_18_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_18_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_18_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_18_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_18_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_18_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_5_modules_18_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_5_modules_18_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_5_modules_18_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_5_modules_18_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_5_modules_18_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_5_modules_18_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_5_modules_18_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_5_modules_18_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_5_modules_18_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_5_modules_18_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_18_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_18_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_18_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_18_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_18_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_18_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_18_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_19_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_5_modules_19_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_5_modules_19_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_19_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_19_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_19_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_19_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_19_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_19_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_19_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_19_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_5_modules_19_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_5_modules_19_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_19_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_19_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_19_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_19_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_19_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_19_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_19_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_19_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_5_modules_19_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_5_modules_19_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_5_modules_19_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_5_modules_19_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_5_modules_19_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_5_modules_19_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_5_modules_19_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_5_modules_19_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_5_modules_19_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_5_modules_19_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_19_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_19_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_19_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_19_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_19_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_19_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_19_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_20_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_5_modules_20_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_5_modules_20_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_20_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_20_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_20_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_20_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_20_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_20_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_20_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_20_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_5_modules_20_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_5_modules_20_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_20_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_20_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_20_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_20_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_20_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_20_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_20_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_20_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_5_modules_20_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_5_modules_20_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_5_modules_20_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_5_modules_20_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_5_modules_20_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_5_modules_20_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_5_modules_20_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_5_modules_20_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_5_modules_20_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_5_modules_20_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_20_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_20_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_20_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_20_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_20_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_20_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_20_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_21_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_5_modules_21_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_5_modules_21_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_21_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_21_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_21_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_21_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_21_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_21_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_21_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_21_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_5_modules_21_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_5_modules_21_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_21_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_21_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_21_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_21_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_21_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_21_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_21_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_21_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_5_modules_21_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_5_modules_21_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_5_modules_21_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_5_modules_21_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_5_modules_21_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_5_modules_21_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_5_modules_21_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_5_modules_21_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_5_modules_21_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_5_modules_21_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_21_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_21_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_21_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_21_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_21_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_21_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_21_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_22_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_5_modules_22_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_5_modules_22_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_22_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_22_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_22_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_22_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_22_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_22_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_22_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_22_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_5_modules_22_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_5_modules_22_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_22_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_22_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_22_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_22_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_22_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_22_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_22_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_22_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_5_modules_22_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_5_modules_22_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_5_modules_22_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_5_modules_22_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_5_modules_22_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_5_modules_22_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_5_modules_22_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_5_modules_22_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_5_modules_22_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_5_modules_22_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_22_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_22_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_22_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_22_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_22_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_22_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_22_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_23_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_5_modules_23_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_5_modules_23_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_23_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_23_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_23_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_23_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_23_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_23_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_23_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_23_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_5_modules_23_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_5_modules_23_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_23_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_23_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_23_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_23_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_23_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_23_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_23_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_23_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_5_modules_23_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_5_modules_23_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_5_modules_23_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_5_modules_23_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_5_modules_23_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_5_modules_23_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_5_modules_23_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_5_modules_23_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_5_modules_23_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_5_modules_23_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_23_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_23_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_23_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_23_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_23_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_23_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_23_modules_bn3_parameters_bias_
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
            l_self_modules_blocks_modules_0_modules_2_modules_conv_exp_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_0_modules_2_modules_conv_exp_parameters_weight_ = (
            None
        )
        x_16 = torch.nn.functional.batch_norm(
            x_15,
            l_self_modules_blocks_modules_0_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_0_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_0_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_15 = (
            l_self_modules_blocks_modules_0_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_0_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_0_modules_2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_2_modules_bn1_parameters_bias_
        ) = None
        x_17 = torch.nn.functional.silu(x_16, inplace=True)
        x_16 = None
        x_18 = torch.conv2d(
            x_17,
            l_self_modules_blocks_modules_0_modules_2_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_17 = l_self_modules_blocks_modules_0_modules_2_modules_conv_pwl_parameters_weight_ = (None)
        x_19 = torch.nn.functional.batch_norm(
            x_18,
            l_self_modules_blocks_modules_0_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_0_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_0_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_18 = (
            l_self_modules_blocks_modules_0_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_0_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_0_modules_2_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_2_modules_bn2_parameters_bias_
        ) = None
        x_20 = x_19 + x_14
        x_19 = x_14 = None
        x_21 = torch.conv2d(
            x_20,
            l_self_modules_blocks_modules_1_modules_0_modules_conv_exp_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_20 = l_self_modules_blocks_modules_1_modules_0_modules_conv_exp_parameters_weight_ = (None)
        x_22 = torch.nn.functional.batch_norm(
            x_21,
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_21 = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_23 = torch.nn.functional.silu(x_22, inplace=True)
        x_22 = None
        x_24 = torch.conv2d(
            x_23,
            l_self_modules_blocks_modules_1_modules_0_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_23 = l_self_modules_blocks_modules_1_modules_0_modules_conv_pwl_parameters_weight_ = (None)
        x_25 = torch.nn.functional.batch_norm(
            x_24,
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_24 = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_bias_
        ) = None
        x_26 = torch.conv2d(
            x_25,
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
        x_27 = torch.nn.functional.batch_norm(
            x_26,
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_26 = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_bias_
        ) = None
        x_28 = torch.nn.functional.silu(x_27, inplace=True)
        x_27 = None
        x_29 = torch.conv2d(
            x_28,
            l_self_modules_blocks_modules_1_modules_1_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_28 = l_self_modules_blocks_modules_1_modules_1_modules_conv_pwl_parameters_weight_ = (None)
        x_30 = torch.nn.functional.batch_norm(
            x_29,
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_29 = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_bias_
        ) = None
        x_31 = x_30 + x_25
        x_30 = x_25 = None
        x_32 = torch.conv2d(
            x_31,
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
        x_33 = torch.nn.functional.batch_norm(
            x_32,
            l_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_32 = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_bias_
        ) = None
        x_34 = torch.nn.functional.silu(x_33, inplace=True)
        x_33 = None
        x_35 = torch.conv2d(
            x_34,
            l_self_modules_blocks_modules_1_modules_2_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_34 = l_self_modules_blocks_modules_1_modules_2_modules_conv_pwl_parameters_weight_ = (None)
        x_36 = torch.nn.functional.batch_norm(
            x_35,
            l_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_35 = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_bias_
        ) = None
        x_37 = x_36 + x_31
        x_36 = x_31 = None
        x_38 = torch.conv2d(
            x_37,
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
        x_39 = torch.nn.functional.batch_norm(
            x_38,
            l_self_modules_blocks_modules_1_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_38 = (
            l_self_modules_blocks_modules_1_modules_3_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_3_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_3_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_3_modules_bn1_parameters_bias_
        ) = None
        x_40 = torch.nn.functional.silu(x_39, inplace=True)
        x_39 = None
        x_41 = torch.conv2d(
            x_40,
            l_self_modules_blocks_modules_1_modules_3_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_40 = l_self_modules_blocks_modules_1_modules_3_modules_conv_pwl_parameters_weight_ = (None)
        x_42 = torch.nn.functional.batch_norm(
            x_41,
            l_self_modules_blocks_modules_1_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_41 = (
            l_self_modules_blocks_modules_1_modules_3_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_3_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_3_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_3_modules_bn2_parameters_bias_
        ) = None
        x_43 = x_42 + x_37
        x_42 = x_37 = None
        x_44 = torch.conv2d(
            x_43,
            l_self_modules_blocks_modules_1_modules_4_modules_conv_exp_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_1_modules_4_modules_conv_exp_parameters_weight_ = (
            None
        )
        x_45 = torch.nn.functional.batch_norm(
            x_44,
            l_self_modules_blocks_modules_1_modules_4_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_4_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_4_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_4_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_44 = (
            l_self_modules_blocks_modules_1_modules_4_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_4_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_4_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_4_modules_bn1_parameters_bias_
        ) = None
        x_46 = torch.nn.functional.silu(x_45, inplace=True)
        x_45 = None
        x_47 = torch.conv2d(
            x_46,
            l_self_modules_blocks_modules_1_modules_4_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_46 = l_self_modules_blocks_modules_1_modules_4_modules_conv_pwl_parameters_weight_ = (None)
        x_48 = torch.nn.functional.batch_norm(
            x_47,
            l_self_modules_blocks_modules_1_modules_4_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_4_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_4_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_4_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_47 = (
            l_self_modules_blocks_modules_1_modules_4_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_4_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_4_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_4_modules_bn2_parameters_bias_
        ) = None
        x_49 = x_48 + x_43
        x_48 = x_43 = None
        x_50 = torch.conv2d(
            x_49,
            l_self_modules_blocks_modules_2_modules_0_modules_conv_exp_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_49 = l_self_modules_blocks_modules_2_modules_0_modules_conv_exp_parameters_weight_ = (None)
        x_51 = torch.nn.functional.batch_norm(
            x_50,
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_50 = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_52 = torch.nn.functional.silu(x_51, inplace=True)
        x_51 = None
        x_53 = torch.conv2d(
            x_52,
            l_self_modules_blocks_modules_2_modules_0_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_52 = l_self_modules_blocks_modules_2_modules_0_modules_conv_pwl_parameters_weight_ = (None)
        x_54 = torch.nn.functional.batch_norm(
            x_53,
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_53 = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_bias_
        ) = None
        x_55 = torch.conv2d(
            x_54,
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
        x_56 = torch.nn.functional.batch_norm(
            x_55,
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_55 = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_bias_
        ) = None
        x_57 = torch.nn.functional.silu(x_56, inplace=True)
        x_56 = None
        x_58 = torch.conv2d(
            x_57,
            l_self_modules_blocks_modules_2_modules_1_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_57 = l_self_modules_blocks_modules_2_modules_1_modules_conv_pwl_parameters_weight_ = (None)
        x_59 = torch.nn.functional.batch_norm(
            x_58,
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_58 = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_bias_
        ) = None
        x_60 = x_59 + x_54
        x_59 = x_54 = None
        x_61 = torch.conv2d(
            x_60,
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
        x_62 = torch.nn.functional.batch_norm(
            x_61,
            l_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_61 = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_bias_
        ) = None
        x_63 = torch.nn.functional.silu(x_62, inplace=True)
        x_62 = None
        x_64 = torch.conv2d(
            x_63,
            l_self_modules_blocks_modules_2_modules_2_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_63 = l_self_modules_blocks_modules_2_modules_2_modules_conv_pwl_parameters_weight_ = (None)
        x_65 = torch.nn.functional.batch_norm(
            x_64,
            l_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_64 = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_bias_
        ) = None
        x_66 = x_65 + x_60
        x_65 = x_60 = None
        x_67 = torch.conv2d(
            x_66,
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
        x_68 = torch.nn.functional.batch_norm(
            x_67,
            l_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_67 = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_bias_
        ) = None
        x_69 = torch.nn.functional.silu(x_68, inplace=True)
        x_68 = None
        x_70 = torch.conv2d(
            x_69,
            l_self_modules_blocks_modules_2_modules_3_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_69 = l_self_modules_blocks_modules_2_modules_3_modules_conv_pwl_parameters_weight_ = (None)
        x_71 = torch.nn.functional.batch_norm(
            x_70,
            l_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_70 = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_bias_
        ) = None
        x_72 = x_71 + x_66
        x_71 = x_66 = None
        x_73 = torch.conv2d(
            x_72,
            l_self_modules_blocks_modules_2_modules_4_modules_conv_exp_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_2_modules_4_modules_conv_exp_parameters_weight_ = (
            None
        )
        x_74 = torch.nn.functional.batch_norm(
            x_73,
            l_self_modules_blocks_modules_2_modules_4_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_4_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_4_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_4_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_73 = (
            l_self_modules_blocks_modules_2_modules_4_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_4_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_4_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_4_modules_bn1_parameters_bias_
        ) = None
        x_75 = torch.nn.functional.silu(x_74, inplace=True)
        x_74 = None
        x_76 = torch.conv2d(
            x_75,
            l_self_modules_blocks_modules_2_modules_4_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_75 = l_self_modules_blocks_modules_2_modules_4_modules_conv_pwl_parameters_weight_ = (None)
        x_77 = torch.nn.functional.batch_norm(
            x_76,
            l_self_modules_blocks_modules_2_modules_4_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_4_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_4_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_4_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_76 = (
            l_self_modules_blocks_modules_2_modules_4_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_4_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_4_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_4_modules_bn2_parameters_bias_
        ) = None
        x_78 = x_77 + x_72
        x_77 = x_72 = None
        x_79 = torch.conv2d(
            x_78,
            l_self_modules_blocks_modules_3_modules_0_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_78 = (
            l_self_modules_blocks_modules_3_modules_0_modules_conv_pw_parameters_weight_
        ) = None
        x_80 = torch.nn.functional.batch_norm(
            x_79,
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_79 = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_81 = torch.nn.functional.silu(x_80, inplace=True)
        x_80 = None
        x_82 = torch.conv2d(
            x_81,
            l_self_modules_blocks_modules_3_modules_0_modules_conv_dw_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            320,
        )
        x_81 = (
            l_self_modules_blocks_modules_3_modules_0_modules_conv_dw_parameters_weight_
        ) = None
        x_83 = torch.nn.functional.batch_norm(
            x_82,
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_82 = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_bias_
        ) = None
        x_84 = torch.nn.functional.silu(x_83, inplace=True)
        x_83 = None
        x_se = x_84.mean((2, 3), keepdim=True)
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
        x_85 = x_84 * sigmoid
        x_84 = sigmoid = None
        x_86 = torch.conv2d(
            x_85,
            l_self_modules_blocks_modules_3_modules_0_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_85 = l_self_modules_blocks_modules_3_modules_0_modules_conv_pwl_parameters_weight_ = (None)
        x_87 = torch.nn.functional.batch_norm(
            x_86,
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_86 = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_bias_
        ) = None
        x_88 = torch.conv2d(
            x_87,
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
        x_89 = torch.nn.functional.batch_norm(
            x_88,
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_88 = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_bias_
        ) = None
        x_90 = torch.nn.functional.silu(x_89, inplace=True)
        x_89 = None
        x_91 = torch.conv2d(
            x_90,
            l_self_modules_blocks_modules_3_modules_1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            608,
        )
        x_90 = (
            l_self_modules_blocks_modules_3_modules_1_modules_conv_dw_parameters_weight_
        ) = None
        x_92 = torch.nn.functional.batch_norm(
            x_91,
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_91 = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_bias_
        ) = None
        x_93 = torch.nn.functional.silu(x_92, inplace=True)
        x_92 = None
        x_se_4 = x_93.mean((2, 3), keepdim=True)
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
        x_94 = x_93 * sigmoid_1
        x_93 = sigmoid_1 = None
        x_95 = torch.conv2d(
            x_94,
            l_self_modules_blocks_modules_3_modules_1_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_94 = l_self_modules_blocks_modules_3_modules_1_modules_conv_pwl_parameters_weight_ = (None)
        x_96 = torch.nn.functional.batch_norm(
            x_95,
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_95 = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_bias_
        ) = None
        x_97 = x_96 + x_87
        x_96 = x_87 = None
        x_98 = torch.conv2d(
            x_97,
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
        x_99 = torch.nn.functional.batch_norm(
            x_98,
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_98 = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_bias_
        ) = None
        x_100 = torch.nn.functional.silu(x_99, inplace=True)
        x_99 = None
        x_101 = torch.conv2d(
            x_100,
            l_self_modules_blocks_modules_3_modules_2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            608,
        )
        x_100 = (
            l_self_modules_blocks_modules_3_modules_2_modules_conv_dw_parameters_weight_
        ) = None
        x_102 = torch.nn.functional.batch_norm(
            x_101,
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_101 = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_bias_
        ) = None
        x_103 = torch.nn.functional.silu(x_102, inplace=True)
        x_102 = None
        x_se_8 = x_103.mean((2, 3), keepdim=True)
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
        x_104 = x_103 * sigmoid_2
        x_103 = sigmoid_2 = None
        x_105 = torch.conv2d(
            x_104,
            l_self_modules_blocks_modules_3_modules_2_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_104 = l_self_modules_blocks_modules_3_modules_2_modules_conv_pwl_parameters_weight_ = (None)
        x_106 = torch.nn.functional.batch_norm(
            x_105,
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_105 = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_bias_
        ) = None
        x_107 = x_106 + x_97
        x_106 = x_97 = None
        x_108 = torch.conv2d(
            x_107,
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
        x_109 = torch.nn.functional.batch_norm(
            x_108,
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_108 = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_bias_
        ) = None
        x_110 = torch.nn.functional.silu(x_109, inplace=True)
        x_109 = None
        x_111 = torch.conv2d(
            x_110,
            l_self_modules_blocks_modules_3_modules_3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            608,
        )
        x_110 = (
            l_self_modules_blocks_modules_3_modules_3_modules_conv_dw_parameters_weight_
        ) = None
        x_112 = torch.nn.functional.batch_norm(
            x_111,
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_111 = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_bias_
        ) = None
        x_113 = torch.nn.functional.silu(x_112, inplace=True)
        x_112 = None
        x_se_12 = x_113.mean((2, 3), keepdim=True)
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
        x_114 = x_113 * sigmoid_3
        x_113 = sigmoid_3 = None
        x_115 = torch.conv2d(
            x_114,
            l_self_modules_blocks_modules_3_modules_3_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_114 = l_self_modules_blocks_modules_3_modules_3_modules_conv_pwl_parameters_weight_ = (None)
        x_116 = torch.nn.functional.batch_norm(
            x_115,
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_115 = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_bias_
        ) = None
        x_117 = x_116 + x_107
        x_116 = x_107 = None
        x_118 = torch.conv2d(
            x_117,
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
        x_119 = torch.nn.functional.batch_norm(
            x_118,
            l_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_118 = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_bias_
        ) = None
        x_120 = torch.nn.functional.silu(x_119, inplace=True)
        x_119 = None
        x_121 = torch.conv2d(
            x_120,
            l_self_modules_blocks_modules_3_modules_4_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            608,
        )
        x_120 = (
            l_self_modules_blocks_modules_3_modules_4_modules_conv_dw_parameters_weight_
        ) = None
        x_122 = torch.nn.functional.batch_norm(
            x_121,
            l_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_121 = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_bias_
        ) = None
        x_123 = torch.nn.functional.silu(x_122, inplace=True)
        x_122 = None
        x_se_16 = x_123.mean((2, 3), keepdim=True)
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
        x_124 = x_123 * sigmoid_4
        x_123 = sigmoid_4 = None
        x_125 = torch.conv2d(
            x_124,
            l_self_modules_blocks_modules_3_modules_4_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_124 = l_self_modules_blocks_modules_3_modules_4_modules_conv_pwl_parameters_weight_ = (None)
        x_126 = torch.nn.functional.batch_norm(
            x_125,
            l_self_modules_blocks_modules_3_modules_4_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_125 = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn3_parameters_bias_
        ) = None
        x_127 = x_126 + x_117
        x_126 = x_117 = None
        x_128 = torch.conv2d(
            x_127,
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
        x_129 = torch.nn.functional.batch_norm(
            x_128,
            l_self_modules_blocks_modules_3_modules_5_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_5_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_5_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_5_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_128 = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn1_parameters_bias_
        ) = None
        x_130 = torch.nn.functional.silu(x_129, inplace=True)
        x_129 = None
        x_131 = torch.conv2d(
            x_130,
            l_self_modules_blocks_modules_3_modules_5_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            608,
        )
        x_130 = (
            l_self_modules_blocks_modules_3_modules_5_modules_conv_dw_parameters_weight_
        ) = None
        x_132 = torch.nn.functional.batch_norm(
            x_131,
            l_self_modules_blocks_modules_3_modules_5_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_5_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_5_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_5_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_131 = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn2_parameters_bias_
        ) = None
        x_133 = torch.nn.functional.silu(x_132, inplace=True)
        x_132 = None
        x_se_20 = x_133.mean((2, 3), keepdim=True)
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
        x_134 = x_133 * sigmoid_5
        x_133 = sigmoid_5 = None
        x_135 = torch.conv2d(
            x_134,
            l_self_modules_blocks_modules_3_modules_5_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_134 = l_self_modules_blocks_modules_3_modules_5_modules_conv_pwl_parameters_weight_ = (None)
        x_136 = torch.nn.functional.batch_norm(
            x_135,
            l_self_modules_blocks_modules_3_modules_5_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_5_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_5_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_5_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_135 = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn3_parameters_bias_
        ) = None
        x_137 = x_136 + x_127
        x_136 = x_127 = None
        x_138 = torch.conv2d(
            x_137,
            l_self_modules_blocks_modules_3_modules_6_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_3_modules_6_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_139 = torch.nn.functional.batch_norm(
            x_138,
            l_self_modules_blocks_modules_3_modules_6_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_6_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_6_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_6_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_138 = (
            l_self_modules_blocks_modules_3_modules_6_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_6_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_6_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_6_modules_bn1_parameters_bias_
        ) = None
        x_140 = torch.nn.functional.silu(x_139, inplace=True)
        x_139 = None
        x_141 = torch.conv2d(
            x_140,
            l_self_modules_blocks_modules_3_modules_6_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            608,
        )
        x_140 = (
            l_self_modules_blocks_modules_3_modules_6_modules_conv_dw_parameters_weight_
        ) = None
        x_142 = torch.nn.functional.batch_norm(
            x_141,
            l_self_modules_blocks_modules_3_modules_6_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_6_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_6_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_6_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_141 = (
            l_self_modules_blocks_modules_3_modules_6_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_6_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_6_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_6_modules_bn2_parameters_bias_
        ) = None
        x_143 = torch.nn.functional.silu(x_142, inplace=True)
        x_142 = None
        x_se_24 = x_143.mean((2, 3), keepdim=True)
        x_se_25 = torch.conv2d(
            x_se_24,
            l_self_modules_blocks_modules_3_modules_6_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_6_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_24 = l_self_modules_blocks_modules_3_modules_6_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_3_modules_6_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_26 = torch.nn.functional.silu(x_se_25, inplace=True)
        x_se_25 = None
        x_se_27 = torch.conv2d(
            x_se_26,
            l_self_modules_blocks_modules_3_modules_6_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_6_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_26 = l_self_modules_blocks_modules_3_modules_6_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_3_modules_6_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_6 = torch.sigmoid(x_se_27)
        x_se_27 = None
        x_144 = x_143 * sigmoid_6
        x_143 = sigmoid_6 = None
        x_145 = torch.conv2d(
            x_144,
            l_self_modules_blocks_modules_3_modules_6_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_144 = l_self_modules_blocks_modules_3_modules_6_modules_conv_pwl_parameters_weight_ = (None)
        x_146 = torch.nn.functional.batch_norm(
            x_145,
            l_self_modules_blocks_modules_3_modules_6_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_6_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_6_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_6_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_145 = (
            l_self_modules_blocks_modules_3_modules_6_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_6_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_6_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_6_modules_bn3_parameters_bias_
        ) = None
        x_147 = x_146 + x_137
        x_146 = x_137 = None
        x_148 = torch.conv2d(
            x_147,
            l_self_modules_blocks_modules_3_modules_7_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_3_modules_7_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_149 = torch.nn.functional.batch_norm(
            x_148,
            l_self_modules_blocks_modules_3_modules_7_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_7_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_7_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_7_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_148 = (
            l_self_modules_blocks_modules_3_modules_7_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_7_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_7_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_7_modules_bn1_parameters_bias_
        ) = None
        x_150 = torch.nn.functional.silu(x_149, inplace=True)
        x_149 = None
        x_151 = torch.conv2d(
            x_150,
            l_self_modules_blocks_modules_3_modules_7_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            608,
        )
        x_150 = (
            l_self_modules_blocks_modules_3_modules_7_modules_conv_dw_parameters_weight_
        ) = None
        x_152 = torch.nn.functional.batch_norm(
            x_151,
            l_self_modules_blocks_modules_3_modules_7_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_7_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_7_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_7_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_151 = (
            l_self_modules_blocks_modules_3_modules_7_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_7_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_7_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_7_modules_bn2_parameters_bias_
        ) = None
        x_153 = torch.nn.functional.silu(x_152, inplace=True)
        x_152 = None
        x_se_28 = x_153.mean((2, 3), keepdim=True)
        x_se_29 = torch.conv2d(
            x_se_28,
            l_self_modules_blocks_modules_3_modules_7_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_7_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_28 = l_self_modules_blocks_modules_3_modules_7_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_3_modules_7_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_30 = torch.nn.functional.silu(x_se_29, inplace=True)
        x_se_29 = None
        x_se_31 = torch.conv2d(
            x_se_30,
            l_self_modules_blocks_modules_3_modules_7_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_7_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_30 = l_self_modules_blocks_modules_3_modules_7_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_3_modules_7_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_7 = torch.sigmoid(x_se_31)
        x_se_31 = None
        x_154 = x_153 * sigmoid_7
        x_153 = sigmoid_7 = None
        x_155 = torch.conv2d(
            x_154,
            l_self_modules_blocks_modules_3_modules_7_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_154 = l_self_modules_blocks_modules_3_modules_7_modules_conv_pwl_parameters_weight_ = (None)
        x_156 = torch.nn.functional.batch_norm(
            x_155,
            l_self_modules_blocks_modules_3_modules_7_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_7_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_7_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_7_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_155 = (
            l_self_modules_blocks_modules_3_modules_7_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_7_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_7_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_7_modules_bn3_parameters_bias_
        ) = None
        x_157 = x_156 + x_147
        x_156 = x_147 = None
        x_158 = torch.conv2d(
            x_157,
            l_self_modules_blocks_modules_4_modules_0_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_157 = (
            l_self_modules_blocks_modules_4_modules_0_modules_conv_pw_parameters_weight_
        ) = None
        x_159 = torch.nn.functional.batch_norm(
            x_158,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_158 = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_160 = torch.nn.functional.silu(x_159, inplace=True)
        x_159 = None
        x_161 = torch.conv2d(
            x_160,
            l_self_modules_blocks_modules_4_modules_0_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            912,
        )
        x_160 = (
            l_self_modules_blocks_modules_4_modules_0_modules_conv_dw_parameters_weight_
        ) = None
        x_162 = torch.nn.functional.batch_norm(
            x_161,
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_161 = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_bias_
        ) = None
        x_163 = torch.nn.functional.silu(x_162, inplace=True)
        x_162 = None
        x_se_32 = x_163.mean((2, 3), keepdim=True)
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
        x_164 = x_163 * sigmoid_8
        x_163 = sigmoid_8 = None
        x_165 = torch.conv2d(
            x_164,
            l_self_modules_blocks_modules_4_modules_0_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_164 = l_self_modules_blocks_modules_4_modules_0_modules_conv_pwl_parameters_weight_ = (None)
        x_166 = torch.nn.functional.batch_norm(
            x_165,
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_165 = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_bias_
        ) = None
        x_167 = torch.conv2d(
            x_166,
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
        x_168 = torch.nn.functional.batch_norm(
            x_167,
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_167 = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_bias_
        ) = None
        x_169 = torch.nn.functional.silu(x_168, inplace=True)
        x_168 = None
        x_170 = torch.conv2d(
            x_169,
            l_self_modules_blocks_modules_4_modules_1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1152,
        )
        x_169 = (
            l_self_modules_blocks_modules_4_modules_1_modules_conv_dw_parameters_weight_
        ) = None
        x_171 = torch.nn.functional.batch_norm(
            x_170,
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_170 = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_bias_
        ) = None
        x_172 = torch.nn.functional.silu(x_171, inplace=True)
        x_171 = None
        x_se_36 = x_172.mean((2, 3), keepdim=True)
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
        x_173 = x_172 * sigmoid_9
        x_172 = sigmoid_9 = None
        x_174 = torch.conv2d(
            x_173,
            l_self_modules_blocks_modules_4_modules_1_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_173 = l_self_modules_blocks_modules_4_modules_1_modules_conv_pwl_parameters_weight_ = (None)
        x_175 = torch.nn.functional.batch_norm(
            x_174,
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_174 = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_bias_
        ) = None
        x_176 = x_175 + x_166
        x_175 = x_166 = None
        x_177 = torch.conv2d(
            x_176,
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
        x_178 = torch.nn.functional.batch_norm(
            x_177,
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_177 = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_bias_
        ) = None
        x_179 = torch.nn.functional.silu(x_178, inplace=True)
        x_178 = None
        x_180 = torch.conv2d(
            x_179,
            l_self_modules_blocks_modules_4_modules_2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1152,
        )
        x_179 = (
            l_self_modules_blocks_modules_4_modules_2_modules_conv_dw_parameters_weight_
        ) = None
        x_181 = torch.nn.functional.batch_norm(
            x_180,
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_180 = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_bias_
        ) = None
        x_182 = torch.nn.functional.silu(x_181, inplace=True)
        x_181 = None
        x_se_40 = x_182.mean((2, 3), keepdim=True)
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
        x_183 = x_182 * sigmoid_10
        x_182 = sigmoid_10 = None
        x_184 = torch.conv2d(
            x_183,
            l_self_modules_blocks_modules_4_modules_2_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_183 = l_self_modules_blocks_modules_4_modules_2_modules_conv_pwl_parameters_weight_ = (None)
        x_185 = torch.nn.functional.batch_norm(
            x_184,
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_184 = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_bias_
        ) = None
        x_186 = x_185 + x_176
        x_185 = x_176 = None
        x_187 = torch.conv2d(
            x_186,
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
        x_188 = torch.nn.functional.batch_norm(
            x_187,
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_187 = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_bias_
        ) = None
        x_189 = torch.nn.functional.silu(x_188, inplace=True)
        x_188 = None
        x_190 = torch.conv2d(
            x_189,
            l_self_modules_blocks_modules_4_modules_3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1152,
        )
        x_189 = (
            l_self_modules_blocks_modules_4_modules_3_modules_conv_dw_parameters_weight_
        ) = None
        x_191 = torch.nn.functional.batch_norm(
            x_190,
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_190 = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_bias_
        ) = None
        x_192 = torch.nn.functional.silu(x_191, inplace=True)
        x_191 = None
        x_se_44 = x_192.mean((2, 3), keepdim=True)
        x_se_45 = torch.conv2d(
            x_se_44,
            l_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_44 = l_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_46 = torch.nn.functional.silu(x_se_45, inplace=True)
        x_se_45 = None
        x_se_47 = torch.conv2d(
            x_se_46,
            l_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_46 = l_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_11 = torch.sigmoid(x_se_47)
        x_se_47 = None
        x_193 = x_192 * sigmoid_11
        x_192 = sigmoid_11 = None
        x_194 = torch.conv2d(
            x_193,
            l_self_modules_blocks_modules_4_modules_3_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_193 = l_self_modules_blocks_modules_4_modules_3_modules_conv_pwl_parameters_weight_ = (None)
        x_195 = torch.nn.functional.batch_norm(
            x_194,
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_194 = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_bias_
        ) = None
        x_196 = x_195 + x_186
        x_195 = x_186 = None
        x_197 = torch.conv2d(
            x_196,
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
        x_198 = torch.nn.functional.batch_norm(
            x_197,
            l_self_modules_blocks_modules_4_modules_4_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_4_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_4_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_4_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_197 = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn1_parameters_bias_
        ) = None
        x_199 = torch.nn.functional.silu(x_198, inplace=True)
        x_198 = None
        x_200 = torch.conv2d(
            x_199,
            l_self_modules_blocks_modules_4_modules_4_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1152,
        )
        x_199 = (
            l_self_modules_blocks_modules_4_modules_4_modules_conv_dw_parameters_weight_
        ) = None
        x_201 = torch.nn.functional.batch_norm(
            x_200,
            l_self_modules_blocks_modules_4_modules_4_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_4_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_4_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_4_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_200 = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn2_parameters_bias_
        ) = None
        x_202 = torch.nn.functional.silu(x_201, inplace=True)
        x_201 = None
        x_se_48 = x_202.mean((2, 3), keepdim=True)
        x_se_49 = torch.conv2d(
            x_se_48,
            l_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_48 = l_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_50 = torch.nn.functional.silu(x_se_49, inplace=True)
        x_se_49 = None
        x_se_51 = torch.conv2d(
            x_se_50,
            l_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_50 = l_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_12 = torch.sigmoid(x_se_51)
        x_se_51 = None
        x_203 = x_202 * sigmoid_12
        x_202 = sigmoid_12 = None
        x_204 = torch.conv2d(
            x_203,
            l_self_modules_blocks_modules_4_modules_4_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_203 = l_self_modules_blocks_modules_4_modules_4_modules_conv_pwl_parameters_weight_ = (None)
        x_205 = torch.nn.functional.batch_norm(
            x_204,
            l_self_modules_blocks_modules_4_modules_4_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_4_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_4_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_4_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_204 = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn3_parameters_bias_
        ) = None
        x_206 = x_205 + x_196
        x_205 = x_196 = None
        x_207 = torch.conv2d(
            x_206,
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
        x_208 = torch.nn.functional.batch_norm(
            x_207,
            l_self_modules_blocks_modules_4_modules_5_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_5_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_5_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_5_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_207 = (
            l_self_modules_blocks_modules_4_modules_5_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_5_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_5_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_5_modules_bn1_parameters_bias_
        ) = None
        x_209 = torch.nn.functional.silu(x_208, inplace=True)
        x_208 = None
        x_210 = torch.conv2d(
            x_209,
            l_self_modules_blocks_modules_4_modules_5_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1152,
        )
        x_209 = (
            l_self_modules_blocks_modules_4_modules_5_modules_conv_dw_parameters_weight_
        ) = None
        x_211 = torch.nn.functional.batch_norm(
            x_210,
            l_self_modules_blocks_modules_4_modules_5_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_5_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_5_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_5_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_210 = (
            l_self_modules_blocks_modules_4_modules_5_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_5_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_5_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_5_modules_bn2_parameters_bias_
        ) = None
        x_212 = torch.nn.functional.silu(x_211, inplace=True)
        x_211 = None
        x_se_52 = x_212.mean((2, 3), keepdim=True)
        x_se_53 = torch.conv2d(
            x_se_52,
            l_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_52 = l_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_54 = torch.nn.functional.silu(x_se_53, inplace=True)
        x_se_53 = None
        x_se_55 = torch.conv2d(
            x_se_54,
            l_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_54 = l_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_13 = torch.sigmoid(x_se_55)
        x_se_55 = None
        x_213 = x_212 * sigmoid_13
        x_212 = sigmoid_13 = None
        x_214 = torch.conv2d(
            x_213,
            l_self_modules_blocks_modules_4_modules_5_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_213 = l_self_modules_blocks_modules_4_modules_5_modules_conv_pwl_parameters_weight_ = (None)
        x_215 = torch.nn.functional.batch_norm(
            x_214,
            l_self_modules_blocks_modules_4_modules_5_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_5_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_5_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_5_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_214 = (
            l_self_modules_blocks_modules_4_modules_5_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_5_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_5_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_5_modules_bn3_parameters_bias_
        ) = None
        x_216 = x_215 + x_206
        x_215 = x_206 = None
        x_217 = torch.conv2d(
            x_216,
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
        x_218 = torch.nn.functional.batch_norm(
            x_217,
            l_self_modules_blocks_modules_4_modules_6_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_6_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_6_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_6_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_217 = (
            l_self_modules_blocks_modules_4_modules_6_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_6_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_6_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_6_modules_bn1_parameters_bias_
        ) = None
        x_219 = torch.nn.functional.silu(x_218, inplace=True)
        x_218 = None
        x_220 = torch.conv2d(
            x_219,
            l_self_modules_blocks_modules_4_modules_6_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1152,
        )
        x_219 = (
            l_self_modules_blocks_modules_4_modules_6_modules_conv_dw_parameters_weight_
        ) = None
        x_221 = torch.nn.functional.batch_norm(
            x_220,
            l_self_modules_blocks_modules_4_modules_6_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_6_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_6_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_6_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_220 = (
            l_self_modules_blocks_modules_4_modules_6_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_6_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_6_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_6_modules_bn2_parameters_bias_
        ) = None
        x_222 = torch.nn.functional.silu(x_221, inplace=True)
        x_221 = None
        x_se_56 = x_222.mean((2, 3), keepdim=True)
        x_se_57 = torch.conv2d(
            x_se_56,
            l_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_56 = l_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_58 = torch.nn.functional.silu(x_se_57, inplace=True)
        x_se_57 = None
        x_se_59 = torch.conv2d(
            x_se_58,
            l_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_58 = l_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_14 = torch.sigmoid(x_se_59)
        x_se_59 = None
        x_223 = x_222 * sigmoid_14
        x_222 = sigmoid_14 = None
        x_224 = torch.conv2d(
            x_223,
            l_self_modules_blocks_modules_4_modules_6_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_223 = l_self_modules_blocks_modules_4_modules_6_modules_conv_pwl_parameters_weight_ = (None)
        x_225 = torch.nn.functional.batch_norm(
            x_224,
            l_self_modules_blocks_modules_4_modules_6_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_6_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_6_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_6_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_224 = (
            l_self_modules_blocks_modules_4_modules_6_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_6_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_6_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_6_modules_bn3_parameters_bias_
        ) = None
        x_226 = x_225 + x_216
        x_225 = x_216 = None
        x_227 = torch.conv2d(
            x_226,
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
        x_228 = torch.nn.functional.batch_norm(
            x_227,
            l_self_modules_blocks_modules_4_modules_7_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_7_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_7_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_7_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_227 = (
            l_self_modules_blocks_modules_4_modules_7_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_7_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_7_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_7_modules_bn1_parameters_bias_
        ) = None
        x_229 = torch.nn.functional.silu(x_228, inplace=True)
        x_228 = None
        x_230 = torch.conv2d(
            x_229,
            l_self_modules_blocks_modules_4_modules_7_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1152,
        )
        x_229 = (
            l_self_modules_blocks_modules_4_modules_7_modules_conv_dw_parameters_weight_
        ) = None
        x_231 = torch.nn.functional.batch_norm(
            x_230,
            l_self_modules_blocks_modules_4_modules_7_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_7_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_7_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_7_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_230 = (
            l_self_modules_blocks_modules_4_modules_7_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_7_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_7_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_7_modules_bn2_parameters_bias_
        ) = None
        x_232 = torch.nn.functional.silu(x_231, inplace=True)
        x_231 = None
        x_se_60 = x_232.mean((2, 3), keepdim=True)
        x_se_61 = torch.conv2d(
            x_se_60,
            l_self_modules_blocks_modules_4_modules_7_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_7_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_60 = l_self_modules_blocks_modules_4_modules_7_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_4_modules_7_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_62 = torch.nn.functional.silu(x_se_61, inplace=True)
        x_se_61 = None
        x_se_63 = torch.conv2d(
            x_se_62,
            l_self_modules_blocks_modules_4_modules_7_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_7_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_62 = l_self_modules_blocks_modules_4_modules_7_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_4_modules_7_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_15 = torch.sigmoid(x_se_63)
        x_se_63 = None
        x_233 = x_232 * sigmoid_15
        x_232 = sigmoid_15 = None
        x_234 = torch.conv2d(
            x_233,
            l_self_modules_blocks_modules_4_modules_7_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_233 = l_self_modules_blocks_modules_4_modules_7_modules_conv_pwl_parameters_weight_ = (None)
        x_235 = torch.nn.functional.batch_norm(
            x_234,
            l_self_modules_blocks_modules_4_modules_7_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_7_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_7_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_7_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_234 = (
            l_self_modules_blocks_modules_4_modules_7_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_7_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_7_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_7_modules_bn3_parameters_bias_
        ) = None
        x_236 = x_235 + x_226
        x_235 = x_226 = None
        x_237 = torch.conv2d(
            x_236,
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
        x_238 = torch.nn.functional.batch_norm(
            x_237,
            l_self_modules_blocks_modules_4_modules_8_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_8_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_8_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_8_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_237 = (
            l_self_modules_blocks_modules_4_modules_8_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_8_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_8_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_8_modules_bn1_parameters_bias_
        ) = None
        x_239 = torch.nn.functional.silu(x_238, inplace=True)
        x_238 = None
        x_240 = torch.conv2d(
            x_239,
            l_self_modules_blocks_modules_4_modules_8_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1152,
        )
        x_239 = (
            l_self_modules_blocks_modules_4_modules_8_modules_conv_dw_parameters_weight_
        ) = None
        x_241 = torch.nn.functional.batch_norm(
            x_240,
            l_self_modules_blocks_modules_4_modules_8_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_8_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_8_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_8_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_240 = (
            l_self_modules_blocks_modules_4_modules_8_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_8_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_8_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_8_modules_bn2_parameters_bias_
        ) = None
        x_242 = torch.nn.functional.silu(x_241, inplace=True)
        x_241 = None
        x_se_64 = x_242.mean((2, 3), keepdim=True)
        x_se_65 = torch.conv2d(
            x_se_64,
            l_self_modules_blocks_modules_4_modules_8_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_8_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_64 = l_self_modules_blocks_modules_4_modules_8_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_4_modules_8_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_66 = torch.nn.functional.silu(x_se_65, inplace=True)
        x_se_65 = None
        x_se_67 = torch.conv2d(
            x_se_66,
            l_self_modules_blocks_modules_4_modules_8_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_8_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_66 = l_self_modules_blocks_modules_4_modules_8_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_4_modules_8_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_16 = torch.sigmoid(x_se_67)
        x_se_67 = None
        x_243 = x_242 * sigmoid_16
        x_242 = sigmoid_16 = None
        x_244 = torch.conv2d(
            x_243,
            l_self_modules_blocks_modules_4_modules_8_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_243 = l_self_modules_blocks_modules_4_modules_8_modules_conv_pwl_parameters_weight_ = (None)
        x_245 = torch.nn.functional.batch_norm(
            x_244,
            l_self_modules_blocks_modules_4_modules_8_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_8_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_8_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_8_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_244 = (
            l_self_modules_blocks_modules_4_modules_8_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_8_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_8_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_8_modules_bn3_parameters_bias_
        ) = None
        x_246 = x_245 + x_236
        x_245 = x_236 = None
        x_247 = torch.conv2d(
            x_246,
            l_self_modules_blocks_modules_4_modules_9_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_4_modules_9_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_248 = torch.nn.functional.batch_norm(
            x_247,
            l_self_modules_blocks_modules_4_modules_9_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_9_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_9_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_9_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_247 = (
            l_self_modules_blocks_modules_4_modules_9_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_9_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_9_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_9_modules_bn1_parameters_bias_
        ) = None
        x_249 = torch.nn.functional.silu(x_248, inplace=True)
        x_248 = None
        x_250 = torch.conv2d(
            x_249,
            l_self_modules_blocks_modules_4_modules_9_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1152,
        )
        x_249 = (
            l_self_modules_blocks_modules_4_modules_9_modules_conv_dw_parameters_weight_
        ) = None
        x_251 = torch.nn.functional.batch_norm(
            x_250,
            l_self_modules_blocks_modules_4_modules_9_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_9_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_9_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_9_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_250 = (
            l_self_modules_blocks_modules_4_modules_9_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_9_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_9_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_9_modules_bn2_parameters_bias_
        ) = None
        x_252 = torch.nn.functional.silu(x_251, inplace=True)
        x_251 = None
        x_se_68 = x_252.mean((2, 3), keepdim=True)
        x_se_69 = torch.conv2d(
            x_se_68,
            l_self_modules_blocks_modules_4_modules_9_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_9_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_68 = l_self_modules_blocks_modules_4_modules_9_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_4_modules_9_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_70 = torch.nn.functional.silu(x_se_69, inplace=True)
        x_se_69 = None
        x_se_71 = torch.conv2d(
            x_se_70,
            l_self_modules_blocks_modules_4_modules_9_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_9_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_70 = l_self_modules_blocks_modules_4_modules_9_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_4_modules_9_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_17 = torch.sigmoid(x_se_71)
        x_se_71 = None
        x_253 = x_252 * sigmoid_17
        x_252 = sigmoid_17 = None
        x_254 = torch.conv2d(
            x_253,
            l_self_modules_blocks_modules_4_modules_9_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_253 = l_self_modules_blocks_modules_4_modules_9_modules_conv_pwl_parameters_weight_ = (None)
        x_255 = torch.nn.functional.batch_norm(
            x_254,
            l_self_modules_blocks_modules_4_modules_9_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_9_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_9_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_9_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_254 = (
            l_self_modules_blocks_modules_4_modules_9_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_9_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_9_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_9_modules_bn3_parameters_bias_
        ) = None
        x_256 = x_255 + x_246
        x_255 = x_246 = None
        x_257 = torch.conv2d(
            x_256,
            l_self_modules_blocks_modules_4_modules_10_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_4_modules_10_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_258 = torch.nn.functional.batch_norm(
            x_257,
            l_self_modules_blocks_modules_4_modules_10_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_10_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_10_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_10_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_257 = (
            l_self_modules_blocks_modules_4_modules_10_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_10_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_10_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_10_modules_bn1_parameters_bias_
        ) = None
        x_259 = torch.nn.functional.silu(x_258, inplace=True)
        x_258 = None
        x_260 = torch.conv2d(
            x_259,
            l_self_modules_blocks_modules_4_modules_10_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1152,
        )
        x_259 = l_self_modules_blocks_modules_4_modules_10_modules_conv_dw_parameters_weight_ = (None)
        x_261 = torch.nn.functional.batch_norm(
            x_260,
            l_self_modules_blocks_modules_4_modules_10_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_10_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_10_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_10_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_260 = (
            l_self_modules_blocks_modules_4_modules_10_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_10_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_10_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_10_modules_bn2_parameters_bias_
        ) = None
        x_262 = torch.nn.functional.silu(x_261, inplace=True)
        x_261 = None
        x_se_72 = x_262.mean((2, 3), keepdim=True)
        x_se_73 = torch.conv2d(
            x_se_72,
            l_self_modules_blocks_modules_4_modules_10_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_10_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_72 = l_self_modules_blocks_modules_4_modules_10_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_4_modules_10_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_74 = torch.nn.functional.silu(x_se_73, inplace=True)
        x_se_73 = None
        x_se_75 = torch.conv2d(
            x_se_74,
            l_self_modules_blocks_modules_4_modules_10_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_10_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_74 = l_self_modules_blocks_modules_4_modules_10_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_4_modules_10_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_18 = torch.sigmoid(x_se_75)
        x_se_75 = None
        x_263 = x_262 * sigmoid_18
        x_262 = sigmoid_18 = None
        x_264 = torch.conv2d(
            x_263,
            l_self_modules_blocks_modules_4_modules_10_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_263 = l_self_modules_blocks_modules_4_modules_10_modules_conv_pwl_parameters_weight_ = (None)
        x_265 = torch.nn.functional.batch_norm(
            x_264,
            l_self_modules_blocks_modules_4_modules_10_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_10_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_10_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_10_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_264 = (
            l_self_modules_blocks_modules_4_modules_10_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_10_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_10_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_10_modules_bn3_parameters_bias_
        ) = None
        x_266 = x_265 + x_256
        x_265 = x_256 = None
        x_267 = torch.conv2d(
            x_266,
            l_self_modules_blocks_modules_4_modules_11_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_4_modules_11_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_268 = torch.nn.functional.batch_norm(
            x_267,
            l_self_modules_blocks_modules_4_modules_11_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_11_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_11_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_11_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_267 = (
            l_self_modules_blocks_modules_4_modules_11_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_11_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_11_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_11_modules_bn1_parameters_bias_
        ) = None
        x_269 = torch.nn.functional.silu(x_268, inplace=True)
        x_268 = None
        x_270 = torch.conv2d(
            x_269,
            l_self_modules_blocks_modules_4_modules_11_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1152,
        )
        x_269 = l_self_modules_blocks_modules_4_modules_11_modules_conv_dw_parameters_weight_ = (None)
        x_271 = torch.nn.functional.batch_norm(
            x_270,
            l_self_modules_blocks_modules_4_modules_11_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_11_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_11_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_11_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_270 = (
            l_self_modules_blocks_modules_4_modules_11_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_11_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_11_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_11_modules_bn2_parameters_bias_
        ) = None
        x_272 = torch.nn.functional.silu(x_271, inplace=True)
        x_271 = None
        x_se_76 = x_272.mean((2, 3), keepdim=True)
        x_se_77 = torch.conv2d(
            x_se_76,
            l_self_modules_blocks_modules_4_modules_11_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_11_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_76 = l_self_modules_blocks_modules_4_modules_11_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_4_modules_11_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_78 = torch.nn.functional.silu(x_se_77, inplace=True)
        x_se_77 = None
        x_se_79 = torch.conv2d(
            x_se_78,
            l_self_modules_blocks_modules_4_modules_11_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_11_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_78 = l_self_modules_blocks_modules_4_modules_11_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_4_modules_11_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_19 = torch.sigmoid(x_se_79)
        x_se_79 = None
        x_273 = x_272 * sigmoid_19
        x_272 = sigmoid_19 = None
        x_274 = torch.conv2d(
            x_273,
            l_self_modules_blocks_modules_4_modules_11_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_273 = l_self_modules_blocks_modules_4_modules_11_modules_conv_pwl_parameters_weight_ = (None)
        x_275 = torch.nn.functional.batch_norm(
            x_274,
            l_self_modules_blocks_modules_4_modules_11_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_11_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_11_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_11_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_274 = (
            l_self_modules_blocks_modules_4_modules_11_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_11_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_11_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_11_modules_bn3_parameters_bias_
        ) = None
        x_276 = x_275 + x_266
        x_275 = x_266 = None
        x_277 = torch.conv2d(
            x_276,
            l_self_modules_blocks_modules_4_modules_12_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_4_modules_12_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_278 = torch.nn.functional.batch_norm(
            x_277,
            l_self_modules_blocks_modules_4_modules_12_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_12_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_12_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_12_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_277 = (
            l_self_modules_blocks_modules_4_modules_12_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_12_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_12_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_12_modules_bn1_parameters_bias_
        ) = None
        x_279 = torch.nn.functional.silu(x_278, inplace=True)
        x_278 = None
        x_280 = torch.conv2d(
            x_279,
            l_self_modules_blocks_modules_4_modules_12_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1152,
        )
        x_279 = l_self_modules_blocks_modules_4_modules_12_modules_conv_dw_parameters_weight_ = (None)
        x_281 = torch.nn.functional.batch_norm(
            x_280,
            l_self_modules_blocks_modules_4_modules_12_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_12_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_12_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_12_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_280 = (
            l_self_modules_blocks_modules_4_modules_12_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_12_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_12_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_12_modules_bn2_parameters_bias_
        ) = None
        x_282 = torch.nn.functional.silu(x_281, inplace=True)
        x_281 = None
        x_se_80 = x_282.mean((2, 3), keepdim=True)
        x_se_81 = torch.conv2d(
            x_se_80,
            l_self_modules_blocks_modules_4_modules_12_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_12_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_80 = l_self_modules_blocks_modules_4_modules_12_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_4_modules_12_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_82 = torch.nn.functional.silu(x_se_81, inplace=True)
        x_se_81 = None
        x_se_83 = torch.conv2d(
            x_se_82,
            l_self_modules_blocks_modules_4_modules_12_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_12_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_82 = l_self_modules_blocks_modules_4_modules_12_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_4_modules_12_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_20 = torch.sigmoid(x_se_83)
        x_se_83 = None
        x_283 = x_282 * sigmoid_20
        x_282 = sigmoid_20 = None
        x_284 = torch.conv2d(
            x_283,
            l_self_modules_blocks_modules_4_modules_12_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_283 = l_self_modules_blocks_modules_4_modules_12_modules_conv_pwl_parameters_weight_ = (None)
        x_285 = torch.nn.functional.batch_norm(
            x_284,
            l_self_modules_blocks_modules_4_modules_12_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_12_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_12_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_12_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_284 = (
            l_self_modules_blocks_modules_4_modules_12_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_12_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_12_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_12_modules_bn3_parameters_bias_
        ) = None
        x_286 = x_285 + x_276
        x_285 = x_276 = None
        x_287 = torch.conv2d(
            x_286,
            l_self_modules_blocks_modules_4_modules_13_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_4_modules_13_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_288 = torch.nn.functional.batch_norm(
            x_287,
            l_self_modules_blocks_modules_4_modules_13_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_13_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_13_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_13_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_287 = (
            l_self_modules_blocks_modules_4_modules_13_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_13_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_13_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_13_modules_bn1_parameters_bias_
        ) = None
        x_289 = torch.nn.functional.silu(x_288, inplace=True)
        x_288 = None
        x_290 = torch.conv2d(
            x_289,
            l_self_modules_blocks_modules_4_modules_13_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1152,
        )
        x_289 = l_self_modules_blocks_modules_4_modules_13_modules_conv_dw_parameters_weight_ = (None)
        x_291 = torch.nn.functional.batch_norm(
            x_290,
            l_self_modules_blocks_modules_4_modules_13_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_13_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_13_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_13_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_290 = (
            l_self_modules_blocks_modules_4_modules_13_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_13_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_13_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_13_modules_bn2_parameters_bias_
        ) = None
        x_292 = torch.nn.functional.silu(x_291, inplace=True)
        x_291 = None
        x_se_84 = x_292.mean((2, 3), keepdim=True)
        x_se_85 = torch.conv2d(
            x_se_84,
            l_self_modules_blocks_modules_4_modules_13_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_13_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_84 = l_self_modules_blocks_modules_4_modules_13_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_4_modules_13_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_86 = torch.nn.functional.silu(x_se_85, inplace=True)
        x_se_85 = None
        x_se_87 = torch.conv2d(
            x_se_86,
            l_self_modules_blocks_modules_4_modules_13_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_13_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_86 = l_self_modules_blocks_modules_4_modules_13_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_4_modules_13_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_21 = torch.sigmoid(x_se_87)
        x_se_87 = None
        x_293 = x_292 * sigmoid_21
        x_292 = sigmoid_21 = None
        x_294 = torch.conv2d(
            x_293,
            l_self_modules_blocks_modules_4_modules_13_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_293 = l_self_modules_blocks_modules_4_modules_13_modules_conv_pwl_parameters_weight_ = (None)
        x_295 = torch.nn.functional.batch_norm(
            x_294,
            l_self_modules_blocks_modules_4_modules_13_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_13_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_13_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_13_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_294 = (
            l_self_modules_blocks_modules_4_modules_13_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_13_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_13_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_13_modules_bn3_parameters_bias_
        ) = None
        x_296 = x_295 + x_286
        x_295 = x_286 = None
        x_297 = torch.conv2d(
            x_296,
            l_self_modules_blocks_modules_4_modules_14_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_4_modules_14_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_298 = torch.nn.functional.batch_norm(
            x_297,
            l_self_modules_blocks_modules_4_modules_14_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_14_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_14_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_14_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_297 = (
            l_self_modules_blocks_modules_4_modules_14_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_14_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_14_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_14_modules_bn1_parameters_bias_
        ) = None
        x_299 = torch.nn.functional.silu(x_298, inplace=True)
        x_298 = None
        x_300 = torch.conv2d(
            x_299,
            l_self_modules_blocks_modules_4_modules_14_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1152,
        )
        x_299 = l_self_modules_blocks_modules_4_modules_14_modules_conv_dw_parameters_weight_ = (None)
        x_301 = torch.nn.functional.batch_norm(
            x_300,
            l_self_modules_blocks_modules_4_modules_14_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_14_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_14_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_14_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_300 = (
            l_self_modules_blocks_modules_4_modules_14_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_14_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_14_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_14_modules_bn2_parameters_bias_
        ) = None
        x_302 = torch.nn.functional.silu(x_301, inplace=True)
        x_301 = None
        x_se_88 = x_302.mean((2, 3), keepdim=True)
        x_se_89 = torch.conv2d(
            x_se_88,
            l_self_modules_blocks_modules_4_modules_14_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_14_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_88 = l_self_modules_blocks_modules_4_modules_14_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_4_modules_14_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_90 = torch.nn.functional.silu(x_se_89, inplace=True)
        x_se_89 = None
        x_se_91 = torch.conv2d(
            x_se_90,
            l_self_modules_blocks_modules_4_modules_14_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_14_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_90 = l_self_modules_blocks_modules_4_modules_14_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_4_modules_14_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_22 = torch.sigmoid(x_se_91)
        x_se_91 = None
        x_303 = x_302 * sigmoid_22
        x_302 = sigmoid_22 = None
        x_304 = torch.conv2d(
            x_303,
            l_self_modules_blocks_modules_4_modules_14_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_303 = l_self_modules_blocks_modules_4_modules_14_modules_conv_pwl_parameters_weight_ = (None)
        x_305 = torch.nn.functional.batch_norm(
            x_304,
            l_self_modules_blocks_modules_4_modules_14_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_14_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_14_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_14_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_304 = (
            l_self_modules_blocks_modules_4_modules_14_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_14_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_14_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_14_modules_bn3_parameters_bias_
        ) = None
        x_306 = x_305 + x_296
        x_305 = x_296 = None
        x_307 = torch.conv2d(
            x_306,
            l_self_modules_blocks_modules_5_modules_0_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_306 = (
            l_self_modules_blocks_modules_5_modules_0_modules_conv_pw_parameters_weight_
        ) = None
        x_308 = torch.nn.functional.batch_norm(
            x_307,
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_307 = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_309 = torch.nn.functional.silu(x_308, inplace=True)
        x_308 = None
        x_310 = torch.conv2d(
            x_309,
            l_self_modules_blocks_modules_5_modules_0_modules_conv_dw_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1152,
        )
        x_309 = (
            l_self_modules_blocks_modules_5_modules_0_modules_conv_dw_parameters_weight_
        ) = None
        x_311 = torch.nn.functional.batch_norm(
            x_310,
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_310 = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_bias_
        ) = None
        x_312 = torch.nn.functional.silu(x_311, inplace=True)
        x_311 = None
        x_se_92 = x_312.mean((2, 3), keepdim=True)
        x_se_93 = torch.conv2d(
            x_se_92,
            l_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_92 = l_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_94 = torch.nn.functional.silu(x_se_93, inplace=True)
        x_se_93 = None
        x_se_95 = torch.conv2d(
            x_se_94,
            l_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_94 = l_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_23 = torch.sigmoid(x_se_95)
        x_se_95 = None
        x_313 = x_312 * sigmoid_23
        x_312 = sigmoid_23 = None
        x_314 = torch.conv2d(
            x_313,
            l_self_modules_blocks_modules_5_modules_0_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_313 = l_self_modules_blocks_modules_5_modules_0_modules_conv_pwl_parameters_weight_ = (None)
        x_315 = torch.nn.functional.batch_norm(
            x_314,
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_314 = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_bias_
        ) = None
        x_316 = torch.conv2d(
            x_315,
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
        x_317 = torch.nn.functional.batch_norm(
            x_316,
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_316 = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_bias_
        ) = None
        x_318 = torch.nn.functional.silu(x_317, inplace=True)
        x_317 = None
        x_319 = torch.conv2d(
            x_318,
            l_self_modules_blocks_modules_5_modules_1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1968,
        )
        x_318 = (
            l_self_modules_blocks_modules_5_modules_1_modules_conv_dw_parameters_weight_
        ) = None
        x_320 = torch.nn.functional.batch_norm(
            x_319,
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_319 = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_bias_
        ) = None
        x_321 = torch.nn.functional.silu(x_320, inplace=True)
        x_320 = None
        x_se_96 = x_321.mean((2, 3), keepdim=True)
        x_se_97 = torch.conv2d(
            x_se_96,
            l_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_96 = l_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_98 = torch.nn.functional.silu(x_se_97, inplace=True)
        x_se_97 = None
        x_se_99 = torch.conv2d(
            x_se_98,
            l_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_98 = l_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_24 = torch.sigmoid(x_se_99)
        x_se_99 = None
        x_322 = x_321 * sigmoid_24
        x_321 = sigmoid_24 = None
        x_323 = torch.conv2d(
            x_322,
            l_self_modules_blocks_modules_5_modules_1_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_322 = l_self_modules_blocks_modules_5_modules_1_modules_conv_pwl_parameters_weight_ = (None)
        x_324 = torch.nn.functional.batch_norm(
            x_323,
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_323 = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_bias_
        ) = None
        x_325 = x_324 + x_315
        x_324 = x_315 = None
        x_326 = torch.conv2d(
            x_325,
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
        x_327 = torch.nn.functional.batch_norm(
            x_326,
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_326 = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_bias_
        ) = None
        x_328 = torch.nn.functional.silu(x_327, inplace=True)
        x_327 = None
        x_329 = torch.conv2d(
            x_328,
            l_self_modules_blocks_modules_5_modules_2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1968,
        )
        x_328 = (
            l_self_modules_blocks_modules_5_modules_2_modules_conv_dw_parameters_weight_
        ) = None
        x_330 = torch.nn.functional.batch_norm(
            x_329,
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_329 = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_bias_
        ) = None
        x_331 = torch.nn.functional.silu(x_330, inplace=True)
        x_330 = None
        x_se_100 = x_331.mean((2, 3), keepdim=True)
        x_se_101 = torch.conv2d(
            x_se_100,
            l_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_100 = l_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_102 = torch.nn.functional.silu(x_se_101, inplace=True)
        x_se_101 = None
        x_se_103 = torch.conv2d(
            x_se_102,
            l_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_102 = l_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_25 = torch.sigmoid(x_se_103)
        x_se_103 = None
        x_332 = x_331 * sigmoid_25
        x_331 = sigmoid_25 = None
        x_333 = torch.conv2d(
            x_332,
            l_self_modules_blocks_modules_5_modules_2_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_332 = l_self_modules_blocks_modules_5_modules_2_modules_conv_pwl_parameters_weight_ = (None)
        x_334 = torch.nn.functional.batch_norm(
            x_333,
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_333 = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_bias_
        ) = None
        x_335 = x_334 + x_325
        x_334 = x_325 = None
        x_336 = torch.conv2d(
            x_335,
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
        x_337 = torch.nn.functional.batch_norm(
            x_336,
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_336 = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_bias_
        ) = None
        x_338 = torch.nn.functional.silu(x_337, inplace=True)
        x_337 = None
        x_339 = torch.conv2d(
            x_338,
            l_self_modules_blocks_modules_5_modules_3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1968,
        )
        x_338 = (
            l_self_modules_blocks_modules_5_modules_3_modules_conv_dw_parameters_weight_
        ) = None
        x_340 = torch.nn.functional.batch_norm(
            x_339,
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_339 = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_bias_
        ) = None
        x_341 = torch.nn.functional.silu(x_340, inplace=True)
        x_340 = None
        x_se_104 = x_341.mean((2, 3), keepdim=True)
        x_se_105 = torch.conv2d(
            x_se_104,
            l_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_104 = l_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_106 = torch.nn.functional.silu(x_se_105, inplace=True)
        x_se_105 = None
        x_se_107 = torch.conv2d(
            x_se_106,
            l_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_106 = l_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_26 = torch.sigmoid(x_se_107)
        x_se_107 = None
        x_342 = x_341 * sigmoid_26
        x_341 = sigmoid_26 = None
        x_343 = torch.conv2d(
            x_342,
            l_self_modules_blocks_modules_5_modules_3_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_342 = l_self_modules_blocks_modules_5_modules_3_modules_conv_pwl_parameters_weight_ = (None)
        x_344 = torch.nn.functional.batch_norm(
            x_343,
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_343 = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_bias_
        ) = None
        x_345 = x_344 + x_335
        x_344 = x_335 = None
        x_346 = torch.conv2d(
            x_345,
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
        x_347 = torch.nn.functional.batch_norm(
            x_346,
            l_self_modules_blocks_modules_5_modules_4_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_4_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_4_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_4_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_346 = (
            l_self_modules_blocks_modules_5_modules_4_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_4_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_4_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_4_modules_bn1_parameters_bias_
        ) = None
        x_348 = torch.nn.functional.silu(x_347, inplace=True)
        x_347 = None
        x_349 = torch.conv2d(
            x_348,
            l_self_modules_blocks_modules_5_modules_4_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1968,
        )
        x_348 = (
            l_self_modules_blocks_modules_5_modules_4_modules_conv_dw_parameters_weight_
        ) = None
        x_350 = torch.nn.functional.batch_norm(
            x_349,
            l_self_modules_blocks_modules_5_modules_4_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_4_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_4_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_4_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_349 = (
            l_self_modules_blocks_modules_5_modules_4_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_4_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_4_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_4_modules_bn2_parameters_bias_
        ) = None
        x_351 = torch.nn.functional.silu(x_350, inplace=True)
        x_350 = None
        x_se_108 = x_351.mean((2, 3), keepdim=True)
        x_se_109 = torch.conv2d(
            x_se_108,
            l_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_108 = l_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_110 = torch.nn.functional.silu(x_se_109, inplace=True)
        x_se_109 = None
        x_se_111 = torch.conv2d(
            x_se_110,
            l_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_110 = l_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_27 = torch.sigmoid(x_se_111)
        x_se_111 = None
        x_352 = x_351 * sigmoid_27
        x_351 = sigmoid_27 = None
        x_353 = torch.conv2d(
            x_352,
            l_self_modules_blocks_modules_5_modules_4_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_352 = l_self_modules_blocks_modules_5_modules_4_modules_conv_pwl_parameters_weight_ = (None)
        x_354 = torch.nn.functional.batch_norm(
            x_353,
            l_self_modules_blocks_modules_5_modules_4_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_4_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_4_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_4_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_353 = (
            l_self_modules_blocks_modules_5_modules_4_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_4_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_4_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_4_modules_bn3_parameters_bias_
        ) = None
        x_355 = x_354 + x_345
        x_354 = x_345 = None
        x_356 = torch.conv2d(
            x_355,
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
        x_357 = torch.nn.functional.batch_norm(
            x_356,
            l_self_modules_blocks_modules_5_modules_5_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_5_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_5_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_5_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_356 = (
            l_self_modules_blocks_modules_5_modules_5_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_5_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_5_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_5_modules_bn1_parameters_bias_
        ) = None
        x_358 = torch.nn.functional.silu(x_357, inplace=True)
        x_357 = None
        x_359 = torch.conv2d(
            x_358,
            l_self_modules_blocks_modules_5_modules_5_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1968,
        )
        x_358 = (
            l_self_modules_blocks_modules_5_modules_5_modules_conv_dw_parameters_weight_
        ) = None
        x_360 = torch.nn.functional.batch_norm(
            x_359,
            l_self_modules_blocks_modules_5_modules_5_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_5_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_5_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_5_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_359 = (
            l_self_modules_blocks_modules_5_modules_5_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_5_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_5_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_5_modules_bn2_parameters_bias_
        ) = None
        x_361 = torch.nn.functional.silu(x_360, inplace=True)
        x_360 = None
        x_se_112 = x_361.mean((2, 3), keepdim=True)
        x_se_113 = torch.conv2d(
            x_se_112,
            l_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_112 = l_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_114 = torch.nn.functional.silu(x_se_113, inplace=True)
        x_se_113 = None
        x_se_115 = torch.conv2d(
            x_se_114,
            l_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_114 = l_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_28 = torch.sigmoid(x_se_115)
        x_se_115 = None
        x_362 = x_361 * sigmoid_28
        x_361 = sigmoid_28 = None
        x_363 = torch.conv2d(
            x_362,
            l_self_modules_blocks_modules_5_modules_5_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_362 = l_self_modules_blocks_modules_5_modules_5_modules_conv_pwl_parameters_weight_ = (None)
        x_364 = torch.nn.functional.batch_norm(
            x_363,
            l_self_modules_blocks_modules_5_modules_5_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_5_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_5_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_5_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_363 = (
            l_self_modules_blocks_modules_5_modules_5_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_5_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_5_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_5_modules_bn3_parameters_bias_
        ) = None
        x_365 = x_364 + x_355
        x_364 = x_355 = None
        x_366 = torch.conv2d(
            x_365,
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
        x_367 = torch.nn.functional.batch_norm(
            x_366,
            l_self_modules_blocks_modules_5_modules_6_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_6_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_6_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_6_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_366 = (
            l_self_modules_blocks_modules_5_modules_6_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_6_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_6_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_6_modules_bn1_parameters_bias_
        ) = None
        x_368 = torch.nn.functional.silu(x_367, inplace=True)
        x_367 = None
        x_369 = torch.conv2d(
            x_368,
            l_self_modules_blocks_modules_5_modules_6_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1968,
        )
        x_368 = (
            l_self_modules_blocks_modules_5_modules_6_modules_conv_dw_parameters_weight_
        ) = None
        x_370 = torch.nn.functional.batch_norm(
            x_369,
            l_self_modules_blocks_modules_5_modules_6_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_6_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_6_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_6_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_369 = (
            l_self_modules_blocks_modules_5_modules_6_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_6_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_6_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_6_modules_bn2_parameters_bias_
        ) = None
        x_371 = torch.nn.functional.silu(x_370, inplace=True)
        x_370 = None
        x_se_116 = x_371.mean((2, 3), keepdim=True)
        x_se_117 = torch.conv2d(
            x_se_116,
            l_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_116 = l_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_118 = torch.nn.functional.silu(x_se_117, inplace=True)
        x_se_117 = None
        x_se_119 = torch.conv2d(
            x_se_118,
            l_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_118 = l_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_29 = torch.sigmoid(x_se_119)
        x_se_119 = None
        x_372 = x_371 * sigmoid_29
        x_371 = sigmoid_29 = None
        x_373 = torch.conv2d(
            x_372,
            l_self_modules_blocks_modules_5_modules_6_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_372 = l_self_modules_blocks_modules_5_modules_6_modules_conv_pwl_parameters_weight_ = (None)
        x_374 = torch.nn.functional.batch_norm(
            x_373,
            l_self_modules_blocks_modules_5_modules_6_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_6_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_6_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_6_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_373 = (
            l_self_modules_blocks_modules_5_modules_6_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_6_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_6_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_6_modules_bn3_parameters_bias_
        ) = None
        x_375 = x_374 + x_365
        x_374 = x_365 = None
        x_376 = torch.conv2d(
            x_375,
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
        x_377 = torch.nn.functional.batch_norm(
            x_376,
            l_self_modules_blocks_modules_5_modules_7_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_7_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_7_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_7_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_376 = (
            l_self_modules_blocks_modules_5_modules_7_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_7_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_7_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_7_modules_bn1_parameters_bias_
        ) = None
        x_378 = torch.nn.functional.silu(x_377, inplace=True)
        x_377 = None
        x_379 = torch.conv2d(
            x_378,
            l_self_modules_blocks_modules_5_modules_7_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1968,
        )
        x_378 = (
            l_self_modules_blocks_modules_5_modules_7_modules_conv_dw_parameters_weight_
        ) = None
        x_380 = torch.nn.functional.batch_norm(
            x_379,
            l_self_modules_blocks_modules_5_modules_7_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_7_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_7_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_7_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_379 = (
            l_self_modules_blocks_modules_5_modules_7_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_7_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_7_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_7_modules_bn2_parameters_bias_
        ) = None
        x_381 = torch.nn.functional.silu(x_380, inplace=True)
        x_380 = None
        x_se_120 = x_381.mean((2, 3), keepdim=True)
        x_se_121 = torch.conv2d(
            x_se_120,
            l_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_120 = l_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_122 = torch.nn.functional.silu(x_se_121, inplace=True)
        x_se_121 = None
        x_se_123 = torch.conv2d(
            x_se_122,
            l_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_122 = l_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_30 = torch.sigmoid(x_se_123)
        x_se_123 = None
        x_382 = x_381 * sigmoid_30
        x_381 = sigmoid_30 = None
        x_383 = torch.conv2d(
            x_382,
            l_self_modules_blocks_modules_5_modules_7_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_382 = l_self_modules_blocks_modules_5_modules_7_modules_conv_pwl_parameters_weight_ = (None)
        x_384 = torch.nn.functional.batch_norm(
            x_383,
            l_self_modules_blocks_modules_5_modules_7_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_7_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_7_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_7_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_383 = (
            l_self_modules_blocks_modules_5_modules_7_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_7_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_7_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_7_modules_bn3_parameters_bias_
        ) = None
        x_385 = x_384 + x_375
        x_384 = x_375 = None
        x_386 = torch.conv2d(
            x_385,
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
        x_387 = torch.nn.functional.batch_norm(
            x_386,
            l_self_modules_blocks_modules_5_modules_8_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_8_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_8_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_8_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_386 = (
            l_self_modules_blocks_modules_5_modules_8_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_8_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_8_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_8_modules_bn1_parameters_bias_
        ) = None
        x_388 = torch.nn.functional.silu(x_387, inplace=True)
        x_387 = None
        x_389 = torch.conv2d(
            x_388,
            l_self_modules_blocks_modules_5_modules_8_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1968,
        )
        x_388 = (
            l_self_modules_blocks_modules_5_modules_8_modules_conv_dw_parameters_weight_
        ) = None
        x_390 = torch.nn.functional.batch_norm(
            x_389,
            l_self_modules_blocks_modules_5_modules_8_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_8_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_8_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_8_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_389 = (
            l_self_modules_blocks_modules_5_modules_8_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_8_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_8_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_8_modules_bn2_parameters_bias_
        ) = None
        x_391 = torch.nn.functional.silu(x_390, inplace=True)
        x_390 = None
        x_se_124 = x_391.mean((2, 3), keepdim=True)
        x_se_125 = torch.conv2d(
            x_se_124,
            l_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_124 = l_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_126 = torch.nn.functional.silu(x_se_125, inplace=True)
        x_se_125 = None
        x_se_127 = torch.conv2d(
            x_se_126,
            l_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_126 = l_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_31 = torch.sigmoid(x_se_127)
        x_se_127 = None
        x_392 = x_391 * sigmoid_31
        x_391 = sigmoid_31 = None
        x_393 = torch.conv2d(
            x_392,
            l_self_modules_blocks_modules_5_modules_8_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_392 = l_self_modules_blocks_modules_5_modules_8_modules_conv_pwl_parameters_weight_ = (None)
        x_394 = torch.nn.functional.batch_norm(
            x_393,
            l_self_modules_blocks_modules_5_modules_8_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_8_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_8_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_8_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_393 = (
            l_self_modules_blocks_modules_5_modules_8_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_8_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_8_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_8_modules_bn3_parameters_bias_
        ) = None
        x_395 = x_394 + x_385
        x_394 = x_385 = None
        x_396 = torch.conv2d(
            x_395,
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
        x_397 = torch.nn.functional.batch_norm(
            x_396,
            l_self_modules_blocks_modules_5_modules_9_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_9_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_9_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_9_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_396 = (
            l_self_modules_blocks_modules_5_modules_9_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_9_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_9_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_9_modules_bn1_parameters_bias_
        ) = None
        x_398 = torch.nn.functional.silu(x_397, inplace=True)
        x_397 = None
        x_399 = torch.conv2d(
            x_398,
            l_self_modules_blocks_modules_5_modules_9_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1968,
        )
        x_398 = (
            l_self_modules_blocks_modules_5_modules_9_modules_conv_dw_parameters_weight_
        ) = None
        x_400 = torch.nn.functional.batch_norm(
            x_399,
            l_self_modules_blocks_modules_5_modules_9_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_9_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_9_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_9_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_399 = (
            l_self_modules_blocks_modules_5_modules_9_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_9_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_9_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_9_modules_bn2_parameters_bias_
        ) = None
        x_401 = torch.nn.functional.silu(x_400, inplace=True)
        x_400 = None
        x_se_128 = x_401.mean((2, 3), keepdim=True)
        x_se_129 = torch.conv2d(
            x_se_128,
            l_self_modules_blocks_modules_5_modules_9_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_9_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_128 = l_self_modules_blocks_modules_5_modules_9_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_9_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_130 = torch.nn.functional.silu(x_se_129, inplace=True)
        x_se_129 = None
        x_se_131 = torch.conv2d(
            x_se_130,
            l_self_modules_blocks_modules_5_modules_9_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_9_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_130 = l_self_modules_blocks_modules_5_modules_9_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_9_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_32 = torch.sigmoid(x_se_131)
        x_se_131 = None
        x_402 = x_401 * sigmoid_32
        x_401 = sigmoid_32 = None
        x_403 = torch.conv2d(
            x_402,
            l_self_modules_blocks_modules_5_modules_9_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_402 = l_self_modules_blocks_modules_5_modules_9_modules_conv_pwl_parameters_weight_ = (None)
        x_404 = torch.nn.functional.batch_norm(
            x_403,
            l_self_modules_blocks_modules_5_modules_9_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_9_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_9_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_9_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_403 = (
            l_self_modules_blocks_modules_5_modules_9_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_9_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_9_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_9_modules_bn3_parameters_bias_
        ) = None
        x_405 = x_404 + x_395
        x_404 = x_395 = None
        x_406 = torch.conv2d(
            x_405,
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
        x_407 = torch.nn.functional.batch_norm(
            x_406,
            l_self_modules_blocks_modules_5_modules_10_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_10_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_10_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_10_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_406 = (
            l_self_modules_blocks_modules_5_modules_10_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_10_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_10_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_10_modules_bn1_parameters_bias_
        ) = None
        x_408 = torch.nn.functional.silu(x_407, inplace=True)
        x_407 = None
        x_409 = torch.conv2d(
            x_408,
            l_self_modules_blocks_modules_5_modules_10_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1968,
        )
        x_408 = l_self_modules_blocks_modules_5_modules_10_modules_conv_dw_parameters_weight_ = (None)
        x_410 = torch.nn.functional.batch_norm(
            x_409,
            l_self_modules_blocks_modules_5_modules_10_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_10_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_10_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_10_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_409 = (
            l_self_modules_blocks_modules_5_modules_10_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_10_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_10_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_10_modules_bn2_parameters_bias_
        ) = None
        x_411 = torch.nn.functional.silu(x_410, inplace=True)
        x_410 = None
        x_se_132 = x_411.mean((2, 3), keepdim=True)
        x_se_133 = torch.conv2d(
            x_se_132,
            l_self_modules_blocks_modules_5_modules_10_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_10_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_132 = l_self_modules_blocks_modules_5_modules_10_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_10_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_134 = torch.nn.functional.silu(x_se_133, inplace=True)
        x_se_133 = None
        x_se_135 = torch.conv2d(
            x_se_134,
            l_self_modules_blocks_modules_5_modules_10_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_10_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_134 = l_self_modules_blocks_modules_5_modules_10_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_10_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_33 = torch.sigmoid(x_se_135)
        x_se_135 = None
        x_412 = x_411 * sigmoid_33
        x_411 = sigmoid_33 = None
        x_413 = torch.conv2d(
            x_412,
            l_self_modules_blocks_modules_5_modules_10_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_412 = l_self_modules_blocks_modules_5_modules_10_modules_conv_pwl_parameters_weight_ = (None)
        x_414 = torch.nn.functional.batch_norm(
            x_413,
            l_self_modules_blocks_modules_5_modules_10_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_10_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_10_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_10_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_413 = (
            l_self_modules_blocks_modules_5_modules_10_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_10_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_10_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_10_modules_bn3_parameters_bias_
        ) = None
        x_415 = x_414 + x_405
        x_414 = x_405 = None
        x_416 = torch.conv2d(
            x_415,
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
        x_417 = torch.nn.functional.batch_norm(
            x_416,
            l_self_modules_blocks_modules_5_modules_11_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_11_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_11_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_11_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_416 = (
            l_self_modules_blocks_modules_5_modules_11_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_11_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_11_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_11_modules_bn1_parameters_bias_
        ) = None
        x_418 = torch.nn.functional.silu(x_417, inplace=True)
        x_417 = None
        x_419 = torch.conv2d(
            x_418,
            l_self_modules_blocks_modules_5_modules_11_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1968,
        )
        x_418 = l_self_modules_blocks_modules_5_modules_11_modules_conv_dw_parameters_weight_ = (None)
        x_420 = torch.nn.functional.batch_norm(
            x_419,
            l_self_modules_blocks_modules_5_modules_11_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_11_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_11_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_11_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_419 = (
            l_self_modules_blocks_modules_5_modules_11_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_11_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_11_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_11_modules_bn2_parameters_bias_
        ) = None
        x_421 = torch.nn.functional.silu(x_420, inplace=True)
        x_420 = None
        x_se_136 = x_421.mean((2, 3), keepdim=True)
        x_se_137 = torch.conv2d(
            x_se_136,
            l_self_modules_blocks_modules_5_modules_11_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_11_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_136 = l_self_modules_blocks_modules_5_modules_11_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_11_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_138 = torch.nn.functional.silu(x_se_137, inplace=True)
        x_se_137 = None
        x_se_139 = torch.conv2d(
            x_se_138,
            l_self_modules_blocks_modules_5_modules_11_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_11_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_138 = l_self_modules_blocks_modules_5_modules_11_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_11_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_34 = torch.sigmoid(x_se_139)
        x_se_139 = None
        x_422 = x_421 * sigmoid_34
        x_421 = sigmoid_34 = None
        x_423 = torch.conv2d(
            x_422,
            l_self_modules_blocks_modules_5_modules_11_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_422 = l_self_modules_blocks_modules_5_modules_11_modules_conv_pwl_parameters_weight_ = (None)
        x_424 = torch.nn.functional.batch_norm(
            x_423,
            l_self_modules_blocks_modules_5_modules_11_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_11_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_11_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_11_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_423 = (
            l_self_modules_blocks_modules_5_modules_11_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_11_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_11_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_11_modules_bn3_parameters_bias_
        ) = None
        x_425 = x_424 + x_415
        x_424 = x_415 = None
        x_426 = torch.conv2d(
            x_425,
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
        x_427 = torch.nn.functional.batch_norm(
            x_426,
            l_self_modules_blocks_modules_5_modules_12_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_12_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_12_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_12_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_426 = (
            l_self_modules_blocks_modules_5_modules_12_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_12_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_12_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_12_modules_bn1_parameters_bias_
        ) = None
        x_428 = torch.nn.functional.silu(x_427, inplace=True)
        x_427 = None
        x_429 = torch.conv2d(
            x_428,
            l_self_modules_blocks_modules_5_modules_12_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1968,
        )
        x_428 = l_self_modules_blocks_modules_5_modules_12_modules_conv_dw_parameters_weight_ = (None)
        x_430 = torch.nn.functional.batch_norm(
            x_429,
            l_self_modules_blocks_modules_5_modules_12_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_12_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_12_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_12_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_429 = (
            l_self_modules_blocks_modules_5_modules_12_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_12_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_12_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_12_modules_bn2_parameters_bias_
        ) = None
        x_431 = torch.nn.functional.silu(x_430, inplace=True)
        x_430 = None
        x_se_140 = x_431.mean((2, 3), keepdim=True)
        x_se_141 = torch.conv2d(
            x_se_140,
            l_self_modules_blocks_modules_5_modules_12_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_12_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_140 = l_self_modules_blocks_modules_5_modules_12_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_12_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_142 = torch.nn.functional.silu(x_se_141, inplace=True)
        x_se_141 = None
        x_se_143 = torch.conv2d(
            x_se_142,
            l_self_modules_blocks_modules_5_modules_12_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_12_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_142 = l_self_modules_blocks_modules_5_modules_12_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_12_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_35 = torch.sigmoid(x_se_143)
        x_se_143 = None
        x_432 = x_431 * sigmoid_35
        x_431 = sigmoid_35 = None
        x_433 = torch.conv2d(
            x_432,
            l_self_modules_blocks_modules_5_modules_12_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_432 = l_self_modules_blocks_modules_5_modules_12_modules_conv_pwl_parameters_weight_ = (None)
        x_434 = torch.nn.functional.batch_norm(
            x_433,
            l_self_modules_blocks_modules_5_modules_12_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_12_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_12_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_12_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_433 = (
            l_self_modules_blocks_modules_5_modules_12_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_12_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_12_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_12_modules_bn3_parameters_bias_
        ) = None
        x_435 = x_434 + x_425
        x_434 = x_425 = None
        x_436 = torch.conv2d(
            x_435,
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
        x_437 = torch.nn.functional.batch_norm(
            x_436,
            l_self_modules_blocks_modules_5_modules_13_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_13_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_13_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_13_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_436 = (
            l_self_modules_blocks_modules_5_modules_13_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_13_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_13_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_13_modules_bn1_parameters_bias_
        ) = None
        x_438 = torch.nn.functional.silu(x_437, inplace=True)
        x_437 = None
        x_439 = torch.conv2d(
            x_438,
            l_self_modules_blocks_modules_5_modules_13_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1968,
        )
        x_438 = l_self_modules_blocks_modules_5_modules_13_modules_conv_dw_parameters_weight_ = (None)
        x_440 = torch.nn.functional.batch_norm(
            x_439,
            l_self_modules_blocks_modules_5_modules_13_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_13_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_13_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_13_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_439 = (
            l_self_modules_blocks_modules_5_modules_13_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_13_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_13_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_13_modules_bn2_parameters_bias_
        ) = None
        x_441 = torch.nn.functional.silu(x_440, inplace=True)
        x_440 = None
        x_se_144 = x_441.mean((2, 3), keepdim=True)
        x_se_145 = torch.conv2d(
            x_se_144,
            l_self_modules_blocks_modules_5_modules_13_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_13_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_144 = l_self_modules_blocks_modules_5_modules_13_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_13_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_146 = torch.nn.functional.silu(x_se_145, inplace=True)
        x_se_145 = None
        x_se_147 = torch.conv2d(
            x_se_146,
            l_self_modules_blocks_modules_5_modules_13_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_13_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_146 = l_self_modules_blocks_modules_5_modules_13_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_13_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_36 = torch.sigmoid(x_se_147)
        x_se_147 = None
        x_442 = x_441 * sigmoid_36
        x_441 = sigmoid_36 = None
        x_443 = torch.conv2d(
            x_442,
            l_self_modules_blocks_modules_5_modules_13_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_442 = l_self_modules_blocks_modules_5_modules_13_modules_conv_pwl_parameters_weight_ = (None)
        x_444 = torch.nn.functional.batch_norm(
            x_443,
            l_self_modules_blocks_modules_5_modules_13_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_13_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_13_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_13_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_443 = (
            l_self_modules_blocks_modules_5_modules_13_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_13_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_13_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_13_modules_bn3_parameters_bias_
        ) = None
        x_445 = x_444 + x_435
        x_444 = x_435 = None
        x_446 = torch.conv2d(
            x_445,
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
        x_447 = torch.nn.functional.batch_norm(
            x_446,
            l_self_modules_blocks_modules_5_modules_14_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_14_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_14_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_14_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_446 = (
            l_self_modules_blocks_modules_5_modules_14_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_14_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_14_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_14_modules_bn1_parameters_bias_
        ) = None
        x_448 = torch.nn.functional.silu(x_447, inplace=True)
        x_447 = None
        x_449 = torch.conv2d(
            x_448,
            l_self_modules_blocks_modules_5_modules_14_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1968,
        )
        x_448 = l_self_modules_blocks_modules_5_modules_14_modules_conv_dw_parameters_weight_ = (None)
        x_450 = torch.nn.functional.batch_norm(
            x_449,
            l_self_modules_blocks_modules_5_modules_14_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_14_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_14_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_14_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_449 = (
            l_self_modules_blocks_modules_5_modules_14_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_14_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_14_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_14_modules_bn2_parameters_bias_
        ) = None
        x_451 = torch.nn.functional.silu(x_450, inplace=True)
        x_450 = None
        x_se_148 = x_451.mean((2, 3), keepdim=True)
        x_se_149 = torch.conv2d(
            x_se_148,
            l_self_modules_blocks_modules_5_modules_14_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_14_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_148 = l_self_modules_blocks_modules_5_modules_14_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_14_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_150 = torch.nn.functional.silu(x_se_149, inplace=True)
        x_se_149 = None
        x_se_151 = torch.conv2d(
            x_se_150,
            l_self_modules_blocks_modules_5_modules_14_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_14_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_150 = l_self_modules_blocks_modules_5_modules_14_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_14_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_37 = torch.sigmoid(x_se_151)
        x_se_151 = None
        x_452 = x_451 * sigmoid_37
        x_451 = sigmoid_37 = None
        x_453 = torch.conv2d(
            x_452,
            l_self_modules_blocks_modules_5_modules_14_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_452 = l_self_modules_blocks_modules_5_modules_14_modules_conv_pwl_parameters_weight_ = (None)
        x_454 = torch.nn.functional.batch_norm(
            x_453,
            l_self_modules_blocks_modules_5_modules_14_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_14_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_14_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_14_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_453 = (
            l_self_modules_blocks_modules_5_modules_14_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_14_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_14_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_14_modules_bn3_parameters_bias_
        ) = None
        x_455 = x_454 + x_445
        x_454 = x_445 = None
        x_456 = torch.conv2d(
            x_455,
            l_self_modules_blocks_modules_5_modules_15_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_5_modules_15_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_457 = torch.nn.functional.batch_norm(
            x_456,
            l_self_modules_blocks_modules_5_modules_15_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_15_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_15_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_15_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_456 = (
            l_self_modules_blocks_modules_5_modules_15_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_15_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_15_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_15_modules_bn1_parameters_bias_
        ) = None
        x_458 = torch.nn.functional.silu(x_457, inplace=True)
        x_457 = None
        x_459 = torch.conv2d(
            x_458,
            l_self_modules_blocks_modules_5_modules_15_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1968,
        )
        x_458 = l_self_modules_blocks_modules_5_modules_15_modules_conv_dw_parameters_weight_ = (None)
        x_460 = torch.nn.functional.batch_norm(
            x_459,
            l_self_modules_blocks_modules_5_modules_15_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_15_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_15_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_15_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_459 = (
            l_self_modules_blocks_modules_5_modules_15_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_15_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_15_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_15_modules_bn2_parameters_bias_
        ) = None
        x_461 = torch.nn.functional.silu(x_460, inplace=True)
        x_460 = None
        x_se_152 = x_461.mean((2, 3), keepdim=True)
        x_se_153 = torch.conv2d(
            x_se_152,
            l_self_modules_blocks_modules_5_modules_15_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_15_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_152 = l_self_modules_blocks_modules_5_modules_15_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_15_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_154 = torch.nn.functional.silu(x_se_153, inplace=True)
        x_se_153 = None
        x_se_155 = torch.conv2d(
            x_se_154,
            l_self_modules_blocks_modules_5_modules_15_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_15_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_154 = l_self_modules_blocks_modules_5_modules_15_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_15_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_38 = torch.sigmoid(x_se_155)
        x_se_155 = None
        x_462 = x_461 * sigmoid_38
        x_461 = sigmoid_38 = None
        x_463 = torch.conv2d(
            x_462,
            l_self_modules_blocks_modules_5_modules_15_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_462 = l_self_modules_blocks_modules_5_modules_15_modules_conv_pwl_parameters_weight_ = (None)
        x_464 = torch.nn.functional.batch_norm(
            x_463,
            l_self_modules_blocks_modules_5_modules_15_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_15_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_15_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_15_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_463 = (
            l_self_modules_blocks_modules_5_modules_15_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_15_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_15_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_15_modules_bn3_parameters_bias_
        ) = None
        x_465 = x_464 + x_455
        x_464 = x_455 = None
        x_466 = torch.conv2d(
            x_465,
            l_self_modules_blocks_modules_5_modules_16_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_5_modules_16_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_467 = torch.nn.functional.batch_norm(
            x_466,
            l_self_modules_blocks_modules_5_modules_16_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_16_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_16_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_16_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_466 = (
            l_self_modules_blocks_modules_5_modules_16_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_16_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_16_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_16_modules_bn1_parameters_bias_
        ) = None
        x_468 = torch.nn.functional.silu(x_467, inplace=True)
        x_467 = None
        x_469 = torch.conv2d(
            x_468,
            l_self_modules_blocks_modules_5_modules_16_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1968,
        )
        x_468 = l_self_modules_blocks_modules_5_modules_16_modules_conv_dw_parameters_weight_ = (None)
        x_470 = torch.nn.functional.batch_norm(
            x_469,
            l_self_modules_blocks_modules_5_modules_16_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_16_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_16_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_16_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_469 = (
            l_self_modules_blocks_modules_5_modules_16_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_16_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_16_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_16_modules_bn2_parameters_bias_
        ) = None
        x_471 = torch.nn.functional.silu(x_470, inplace=True)
        x_470 = None
        x_se_156 = x_471.mean((2, 3), keepdim=True)
        x_se_157 = torch.conv2d(
            x_se_156,
            l_self_modules_blocks_modules_5_modules_16_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_16_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_156 = l_self_modules_blocks_modules_5_modules_16_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_16_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_158 = torch.nn.functional.silu(x_se_157, inplace=True)
        x_se_157 = None
        x_se_159 = torch.conv2d(
            x_se_158,
            l_self_modules_blocks_modules_5_modules_16_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_16_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_158 = l_self_modules_blocks_modules_5_modules_16_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_16_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_39 = torch.sigmoid(x_se_159)
        x_se_159 = None
        x_472 = x_471 * sigmoid_39
        x_471 = sigmoid_39 = None
        x_473 = torch.conv2d(
            x_472,
            l_self_modules_blocks_modules_5_modules_16_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_472 = l_self_modules_blocks_modules_5_modules_16_modules_conv_pwl_parameters_weight_ = (None)
        x_474 = torch.nn.functional.batch_norm(
            x_473,
            l_self_modules_blocks_modules_5_modules_16_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_16_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_16_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_16_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_473 = (
            l_self_modules_blocks_modules_5_modules_16_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_16_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_16_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_16_modules_bn3_parameters_bias_
        ) = None
        x_475 = x_474 + x_465
        x_474 = x_465 = None
        x_476 = torch.conv2d(
            x_475,
            l_self_modules_blocks_modules_5_modules_17_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_5_modules_17_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_477 = torch.nn.functional.batch_norm(
            x_476,
            l_self_modules_blocks_modules_5_modules_17_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_17_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_17_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_17_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_476 = (
            l_self_modules_blocks_modules_5_modules_17_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_17_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_17_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_17_modules_bn1_parameters_bias_
        ) = None
        x_478 = torch.nn.functional.silu(x_477, inplace=True)
        x_477 = None
        x_479 = torch.conv2d(
            x_478,
            l_self_modules_blocks_modules_5_modules_17_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1968,
        )
        x_478 = l_self_modules_blocks_modules_5_modules_17_modules_conv_dw_parameters_weight_ = (None)
        x_480 = torch.nn.functional.batch_norm(
            x_479,
            l_self_modules_blocks_modules_5_modules_17_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_17_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_17_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_17_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_479 = (
            l_self_modules_blocks_modules_5_modules_17_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_17_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_17_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_17_modules_bn2_parameters_bias_
        ) = None
        x_481 = torch.nn.functional.silu(x_480, inplace=True)
        x_480 = None
        x_se_160 = x_481.mean((2, 3), keepdim=True)
        x_se_161 = torch.conv2d(
            x_se_160,
            l_self_modules_blocks_modules_5_modules_17_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_17_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_160 = l_self_modules_blocks_modules_5_modules_17_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_17_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_162 = torch.nn.functional.silu(x_se_161, inplace=True)
        x_se_161 = None
        x_se_163 = torch.conv2d(
            x_se_162,
            l_self_modules_blocks_modules_5_modules_17_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_17_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_162 = l_self_modules_blocks_modules_5_modules_17_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_17_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_40 = torch.sigmoid(x_se_163)
        x_se_163 = None
        x_482 = x_481 * sigmoid_40
        x_481 = sigmoid_40 = None
        x_483 = torch.conv2d(
            x_482,
            l_self_modules_blocks_modules_5_modules_17_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_482 = l_self_modules_blocks_modules_5_modules_17_modules_conv_pwl_parameters_weight_ = (None)
        x_484 = torch.nn.functional.batch_norm(
            x_483,
            l_self_modules_blocks_modules_5_modules_17_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_17_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_17_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_17_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_483 = (
            l_self_modules_blocks_modules_5_modules_17_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_17_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_17_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_17_modules_bn3_parameters_bias_
        ) = None
        x_485 = x_484 + x_475
        x_484 = x_475 = None
        x_486 = torch.conv2d(
            x_485,
            l_self_modules_blocks_modules_5_modules_18_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_5_modules_18_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_487 = torch.nn.functional.batch_norm(
            x_486,
            l_self_modules_blocks_modules_5_modules_18_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_18_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_18_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_18_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_486 = (
            l_self_modules_blocks_modules_5_modules_18_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_18_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_18_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_18_modules_bn1_parameters_bias_
        ) = None
        x_488 = torch.nn.functional.silu(x_487, inplace=True)
        x_487 = None
        x_489 = torch.conv2d(
            x_488,
            l_self_modules_blocks_modules_5_modules_18_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1968,
        )
        x_488 = l_self_modules_blocks_modules_5_modules_18_modules_conv_dw_parameters_weight_ = (None)
        x_490 = torch.nn.functional.batch_norm(
            x_489,
            l_self_modules_blocks_modules_5_modules_18_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_18_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_18_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_18_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_489 = (
            l_self_modules_blocks_modules_5_modules_18_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_18_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_18_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_18_modules_bn2_parameters_bias_
        ) = None
        x_491 = torch.nn.functional.silu(x_490, inplace=True)
        x_490 = None
        x_se_164 = x_491.mean((2, 3), keepdim=True)
        x_se_165 = torch.conv2d(
            x_se_164,
            l_self_modules_blocks_modules_5_modules_18_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_18_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_164 = l_self_modules_blocks_modules_5_modules_18_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_18_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_166 = torch.nn.functional.silu(x_se_165, inplace=True)
        x_se_165 = None
        x_se_167 = torch.conv2d(
            x_se_166,
            l_self_modules_blocks_modules_5_modules_18_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_18_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_166 = l_self_modules_blocks_modules_5_modules_18_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_18_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_41 = torch.sigmoid(x_se_167)
        x_se_167 = None
        x_492 = x_491 * sigmoid_41
        x_491 = sigmoid_41 = None
        x_493 = torch.conv2d(
            x_492,
            l_self_modules_blocks_modules_5_modules_18_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_492 = l_self_modules_blocks_modules_5_modules_18_modules_conv_pwl_parameters_weight_ = (None)
        x_494 = torch.nn.functional.batch_norm(
            x_493,
            l_self_modules_blocks_modules_5_modules_18_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_18_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_18_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_18_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_493 = (
            l_self_modules_blocks_modules_5_modules_18_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_18_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_18_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_18_modules_bn3_parameters_bias_
        ) = None
        x_495 = x_494 + x_485
        x_494 = x_485 = None
        x_496 = torch.conv2d(
            x_495,
            l_self_modules_blocks_modules_5_modules_19_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_5_modules_19_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_497 = torch.nn.functional.batch_norm(
            x_496,
            l_self_modules_blocks_modules_5_modules_19_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_19_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_19_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_19_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_496 = (
            l_self_modules_blocks_modules_5_modules_19_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_19_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_19_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_19_modules_bn1_parameters_bias_
        ) = None
        x_498 = torch.nn.functional.silu(x_497, inplace=True)
        x_497 = None
        x_499 = torch.conv2d(
            x_498,
            l_self_modules_blocks_modules_5_modules_19_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1968,
        )
        x_498 = l_self_modules_blocks_modules_5_modules_19_modules_conv_dw_parameters_weight_ = (None)
        x_500 = torch.nn.functional.batch_norm(
            x_499,
            l_self_modules_blocks_modules_5_modules_19_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_19_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_19_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_19_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_499 = (
            l_self_modules_blocks_modules_5_modules_19_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_19_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_19_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_19_modules_bn2_parameters_bias_
        ) = None
        x_501 = torch.nn.functional.silu(x_500, inplace=True)
        x_500 = None
        x_se_168 = x_501.mean((2, 3), keepdim=True)
        x_se_169 = torch.conv2d(
            x_se_168,
            l_self_modules_blocks_modules_5_modules_19_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_19_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_168 = l_self_modules_blocks_modules_5_modules_19_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_19_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_170 = torch.nn.functional.silu(x_se_169, inplace=True)
        x_se_169 = None
        x_se_171 = torch.conv2d(
            x_se_170,
            l_self_modules_blocks_modules_5_modules_19_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_19_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_170 = l_self_modules_blocks_modules_5_modules_19_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_19_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_42 = torch.sigmoid(x_se_171)
        x_se_171 = None
        x_502 = x_501 * sigmoid_42
        x_501 = sigmoid_42 = None
        x_503 = torch.conv2d(
            x_502,
            l_self_modules_blocks_modules_5_modules_19_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_502 = l_self_modules_blocks_modules_5_modules_19_modules_conv_pwl_parameters_weight_ = (None)
        x_504 = torch.nn.functional.batch_norm(
            x_503,
            l_self_modules_blocks_modules_5_modules_19_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_19_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_19_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_19_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_503 = (
            l_self_modules_blocks_modules_5_modules_19_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_19_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_19_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_19_modules_bn3_parameters_bias_
        ) = None
        x_505 = x_504 + x_495
        x_504 = x_495 = None
        x_506 = torch.conv2d(
            x_505,
            l_self_modules_blocks_modules_5_modules_20_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_5_modules_20_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_507 = torch.nn.functional.batch_norm(
            x_506,
            l_self_modules_blocks_modules_5_modules_20_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_20_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_20_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_20_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_506 = (
            l_self_modules_blocks_modules_5_modules_20_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_20_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_20_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_20_modules_bn1_parameters_bias_
        ) = None
        x_508 = torch.nn.functional.silu(x_507, inplace=True)
        x_507 = None
        x_509 = torch.conv2d(
            x_508,
            l_self_modules_blocks_modules_5_modules_20_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1968,
        )
        x_508 = l_self_modules_blocks_modules_5_modules_20_modules_conv_dw_parameters_weight_ = (None)
        x_510 = torch.nn.functional.batch_norm(
            x_509,
            l_self_modules_blocks_modules_5_modules_20_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_20_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_20_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_20_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_509 = (
            l_self_modules_blocks_modules_5_modules_20_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_20_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_20_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_20_modules_bn2_parameters_bias_
        ) = None
        x_511 = torch.nn.functional.silu(x_510, inplace=True)
        x_510 = None
        x_se_172 = x_511.mean((2, 3), keepdim=True)
        x_se_173 = torch.conv2d(
            x_se_172,
            l_self_modules_blocks_modules_5_modules_20_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_20_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_172 = l_self_modules_blocks_modules_5_modules_20_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_20_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_174 = torch.nn.functional.silu(x_se_173, inplace=True)
        x_se_173 = None
        x_se_175 = torch.conv2d(
            x_se_174,
            l_self_modules_blocks_modules_5_modules_20_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_20_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_174 = l_self_modules_blocks_modules_5_modules_20_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_20_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_43 = torch.sigmoid(x_se_175)
        x_se_175 = None
        x_512 = x_511 * sigmoid_43
        x_511 = sigmoid_43 = None
        x_513 = torch.conv2d(
            x_512,
            l_self_modules_blocks_modules_5_modules_20_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_512 = l_self_modules_blocks_modules_5_modules_20_modules_conv_pwl_parameters_weight_ = (None)
        x_514 = torch.nn.functional.batch_norm(
            x_513,
            l_self_modules_blocks_modules_5_modules_20_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_20_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_20_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_20_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_513 = (
            l_self_modules_blocks_modules_5_modules_20_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_20_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_20_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_20_modules_bn3_parameters_bias_
        ) = None
        x_515 = x_514 + x_505
        x_514 = x_505 = None
        x_516 = torch.conv2d(
            x_515,
            l_self_modules_blocks_modules_5_modules_21_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_5_modules_21_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_517 = torch.nn.functional.batch_norm(
            x_516,
            l_self_modules_blocks_modules_5_modules_21_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_21_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_21_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_21_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_516 = (
            l_self_modules_blocks_modules_5_modules_21_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_21_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_21_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_21_modules_bn1_parameters_bias_
        ) = None
        x_518 = torch.nn.functional.silu(x_517, inplace=True)
        x_517 = None
        x_519 = torch.conv2d(
            x_518,
            l_self_modules_blocks_modules_5_modules_21_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1968,
        )
        x_518 = l_self_modules_blocks_modules_5_modules_21_modules_conv_dw_parameters_weight_ = (None)
        x_520 = torch.nn.functional.batch_norm(
            x_519,
            l_self_modules_blocks_modules_5_modules_21_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_21_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_21_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_21_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_519 = (
            l_self_modules_blocks_modules_5_modules_21_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_21_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_21_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_21_modules_bn2_parameters_bias_
        ) = None
        x_521 = torch.nn.functional.silu(x_520, inplace=True)
        x_520 = None
        x_se_176 = x_521.mean((2, 3), keepdim=True)
        x_se_177 = torch.conv2d(
            x_se_176,
            l_self_modules_blocks_modules_5_modules_21_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_21_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_176 = l_self_modules_blocks_modules_5_modules_21_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_21_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_178 = torch.nn.functional.silu(x_se_177, inplace=True)
        x_se_177 = None
        x_se_179 = torch.conv2d(
            x_se_178,
            l_self_modules_blocks_modules_5_modules_21_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_21_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_178 = l_self_modules_blocks_modules_5_modules_21_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_21_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_44 = torch.sigmoid(x_se_179)
        x_se_179 = None
        x_522 = x_521 * sigmoid_44
        x_521 = sigmoid_44 = None
        x_523 = torch.conv2d(
            x_522,
            l_self_modules_blocks_modules_5_modules_21_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_522 = l_self_modules_blocks_modules_5_modules_21_modules_conv_pwl_parameters_weight_ = (None)
        x_524 = torch.nn.functional.batch_norm(
            x_523,
            l_self_modules_blocks_modules_5_modules_21_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_21_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_21_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_21_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_523 = (
            l_self_modules_blocks_modules_5_modules_21_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_21_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_21_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_21_modules_bn3_parameters_bias_
        ) = None
        x_525 = x_524 + x_515
        x_524 = x_515 = None
        x_526 = torch.conv2d(
            x_525,
            l_self_modules_blocks_modules_5_modules_22_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_5_modules_22_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_527 = torch.nn.functional.batch_norm(
            x_526,
            l_self_modules_blocks_modules_5_modules_22_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_22_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_22_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_22_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_526 = (
            l_self_modules_blocks_modules_5_modules_22_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_22_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_22_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_22_modules_bn1_parameters_bias_
        ) = None
        x_528 = torch.nn.functional.silu(x_527, inplace=True)
        x_527 = None
        x_529 = torch.conv2d(
            x_528,
            l_self_modules_blocks_modules_5_modules_22_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1968,
        )
        x_528 = l_self_modules_blocks_modules_5_modules_22_modules_conv_dw_parameters_weight_ = (None)
        x_530 = torch.nn.functional.batch_norm(
            x_529,
            l_self_modules_blocks_modules_5_modules_22_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_22_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_22_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_22_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_529 = (
            l_self_modules_blocks_modules_5_modules_22_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_22_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_22_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_22_modules_bn2_parameters_bias_
        ) = None
        x_531 = torch.nn.functional.silu(x_530, inplace=True)
        x_530 = None
        x_se_180 = x_531.mean((2, 3), keepdim=True)
        x_se_181 = torch.conv2d(
            x_se_180,
            l_self_modules_blocks_modules_5_modules_22_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_22_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_180 = l_self_modules_blocks_modules_5_modules_22_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_22_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_182 = torch.nn.functional.silu(x_se_181, inplace=True)
        x_se_181 = None
        x_se_183 = torch.conv2d(
            x_se_182,
            l_self_modules_blocks_modules_5_modules_22_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_22_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_182 = l_self_modules_blocks_modules_5_modules_22_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_22_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_45 = torch.sigmoid(x_se_183)
        x_se_183 = None
        x_532 = x_531 * sigmoid_45
        x_531 = sigmoid_45 = None
        x_533 = torch.conv2d(
            x_532,
            l_self_modules_blocks_modules_5_modules_22_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_532 = l_self_modules_blocks_modules_5_modules_22_modules_conv_pwl_parameters_weight_ = (None)
        x_534 = torch.nn.functional.batch_norm(
            x_533,
            l_self_modules_blocks_modules_5_modules_22_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_22_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_22_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_22_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_533 = (
            l_self_modules_blocks_modules_5_modules_22_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_22_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_22_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_22_modules_bn3_parameters_bias_
        ) = None
        x_535 = x_534 + x_525
        x_534 = x_525 = None
        x_536 = torch.conv2d(
            x_535,
            l_self_modules_blocks_modules_5_modules_23_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_5_modules_23_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_537 = torch.nn.functional.batch_norm(
            x_536,
            l_self_modules_blocks_modules_5_modules_23_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_23_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_23_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_23_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_536 = (
            l_self_modules_blocks_modules_5_modules_23_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_23_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_23_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_23_modules_bn1_parameters_bias_
        ) = None
        x_538 = torch.nn.functional.silu(x_537, inplace=True)
        x_537 = None
        x_539 = torch.conv2d(
            x_538,
            l_self_modules_blocks_modules_5_modules_23_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1968,
        )
        x_538 = l_self_modules_blocks_modules_5_modules_23_modules_conv_dw_parameters_weight_ = (None)
        x_540 = torch.nn.functional.batch_norm(
            x_539,
            l_self_modules_blocks_modules_5_modules_23_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_23_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_23_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_23_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_539 = (
            l_self_modules_blocks_modules_5_modules_23_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_23_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_23_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_23_modules_bn2_parameters_bias_
        ) = None
        x_541 = torch.nn.functional.silu(x_540, inplace=True)
        x_540 = None
        x_se_184 = x_541.mean((2, 3), keepdim=True)
        x_se_185 = torch.conv2d(
            x_se_184,
            l_self_modules_blocks_modules_5_modules_23_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_23_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_184 = l_self_modules_blocks_modules_5_modules_23_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_23_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_186 = torch.nn.functional.silu(x_se_185, inplace=True)
        x_se_185 = None
        x_se_187 = torch.conv2d(
            x_se_186,
            l_self_modules_blocks_modules_5_modules_23_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_23_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_186 = l_self_modules_blocks_modules_5_modules_23_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_23_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_46 = torch.sigmoid(x_se_187)
        x_se_187 = None
        x_542 = x_541 * sigmoid_46
        x_541 = sigmoid_46 = None
        x_543 = torch.conv2d(
            x_542,
            l_self_modules_blocks_modules_5_modules_23_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_542 = l_self_modules_blocks_modules_5_modules_23_modules_conv_pwl_parameters_weight_ = (None)
        x_544 = torch.nn.functional.batch_norm(
            x_543,
            l_self_modules_blocks_modules_5_modules_23_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_23_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_23_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_23_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_543 = (
            l_self_modules_blocks_modules_5_modules_23_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_23_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_23_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_23_modules_bn3_parameters_bias_
        ) = None
        x_545 = x_544 + x_535
        x_544 = x_535 = None
        x_546 = torch.conv2d(
            x_545,
            l_self_modules_conv_head_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_545 = l_self_modules_conv_head_parameters_weight_ = None
        x_547 = torch.nn.functional.batch_norm(
            x_546,
            l_self_modules_bn2_buffers_running_mean_,
            l_self_modules_bn2_buffers_running_var_,
            l_self_modules_bn2_parameters_weight_,
            l_self_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_546 = (
            l_self_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_bn2_parameters_weight_
        ) = l_self_modules_bn2_parameters_bias_ = None
        x_548 = torch.nn.functional.silu(x_547, inplace=True)
        x_547 = None
        x_549 = torch.nn.functional.adaptive_avg_pool2d(x_548, 1)
        x_548 = None
        x_550 = x_549.flatten(1, -1)
        x_549 = None
        x_551 = torch._C._nn.linear(
            x_550,
            l_self_modules_classifier_parameters_weight_,
            l_self_modules_classifier_parameters_bias_,
        )
        x_550 = (
            l_self_modules_classifier_parameters_weight_
        ) = l_self_modules_classifier_parameters_bias_ = None
        return (x_551,)
