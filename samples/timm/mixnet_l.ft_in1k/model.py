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
        L_self_modules_blocks_modules_0_modules_0_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_conv_pw_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_conv_pw_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_conv_dw_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_conv_dw_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_conv_dw_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_conv_pwl_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_conv_pwl_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_conv_pw_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_conv_pw_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_conv_pwl_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_conv_pwl_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_conv_dw_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_conv_dw_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_conv_dw_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_conv_dw_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_2_modules_1_modules_conv_pw_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_conv_pw_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_conv_dw_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_conv_dw_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_conv_pwl_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_conv_pwl_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_conv_pw_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_conv_pw_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_conv_dw_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_conv_dw_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_conv_pwl_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_conv_pwl_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_2_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_2_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_conv_pw_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_conv_pw_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_conv_dw_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_conv_dw_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_conv_pwl_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_conv_pwl_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_3_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_3_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_conv_dw_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_conv_dw_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_conv_dw_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_3_modules_1_modules_conv_pw_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_conv_pw_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_conv_dw_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_conv_dw_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_conv_dw_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_conv_dw_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_conv_pwl_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_conv_pwl_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_conv_pw_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_conv_pw_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_conv_dw_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_conv_dw_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_conv_dw_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_conv_dw_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_conv_pwl_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_conv_pwl_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_conv_pw_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_conv_pw_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_conv_dw_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_conv_dw_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_conv_dw_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_conv_dw_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_conv_pwl_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_conv_pwl_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_4_modules_1_modules_conv_pw_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_1_modules_conv_pw_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_1_modules_conv_dw_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_1_modules_conv_dw_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_1_modules_conv_dw_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_1_modules_conv_dw_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_1_modules_conv_pwl_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_1_modules_conv_pwl_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_2_modules_conv_pw_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_2_modules_conv_pw_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_2_modules_conv_dw_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_2_modules_conv_dw_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_2_modules_conv_dw_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_2_modules_conv_dw_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_2_modules_conv_pwl_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_2_modules_conv_pwl_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_3_modules_conv_pw_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_3_modules_conv_pw_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_3_modules_conv_dw_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_3_modules_conv_dw_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_3_modules_conv_dw_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_3_modules_conv_dw_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_3_modules_conv_pwl_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_3_modules_conv_pwl_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_conv_dw_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_conv_dw_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_conv_dw_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_conv_dw_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_5_modules_1_modules_conv_dw_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_1_modules_conv_dw_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_1_modules_conv_dw_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_1_modules_conv_dw_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_1_modules_conv_pwl_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_1_modules_conv_pwl_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_2_modules_conv_dw_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_2_modules_conv_dw_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_2_modules_conv_dw_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_2_modules_conv_dw_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_2_modules_conv_pwl_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_2_modules_conv_pwl_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_3_modules_conv_dw_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_3_modules_conv_dw_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_3_modules_conv_dw_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_3_modules_conv_dw_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_3_modules_conv_pwl_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_3_modules_conv_pwl_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_blocks_modules_1_modules_0_modules_conv_pw_modules_0_parameters_weight_ = L_self_modules_blocks_modules_1_modules_0_modules_conv_pw_modules_0_parameters_weight_
        l_self_modules_blocks_modules_1_modules_0_modules_conv_pw_modules_1_parameters_weight_ = L_self_modules_blocks_modules_1_modules_0_modules_conv_pw_modules_1_parameters_weight_
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
        l_self_modules_blocks_modules_1_modules_0_modules_conv_dw_modules_0_parameters_weight_ = L_self_modules_blocks_modules_1_modules_0_modules_conv_dw_modules_0_parameters_weight_
        l_self_modules_blocks_modules_1_modules_0_modules_conv_dw_modules_1_parameters_weight_ = L_self_modules_blocks_modules_1_modules_0_modules_conv_dw_modules_1_parameters_weight_
        l_self_modules_blocks_modules_1_modules_0_modules_conv_dw_modules_2_parameters_weight_ = L_self_modules_blocks_modules_1_modules_0_modules_conv_dw_modules_2_parameters_weight_
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
        l_self_modules_blocks_modules_1_modules_0_modules_conv_pwl_modules_0_parameters_weight_ = L_self_modules_blocks_modules_1_modules_0_modules_conv_pwl_modules_0_parameters_weight_
        l_self_modules_blocks_modules_1_modules_0_modules_conv_pwl_modules_1_parameters_weight_ = L_self_modules_blocks_modules_1_modules_0_modules_conv_pwl_modules_1_parameters_weight_
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
        l_self_modules_blocks_modules_1_modules_1_modules_conv_pw_modules_0_parameters_weight_ = L_self_modules_blocks_modules_1_modules_1_modules_conv_pw_modules_0_parameters_weight_
        l_self_modules_blocks_modules_1_modules_1_modules_conv_pw_modules_1_parameters_weight_ = L_self_modules_blocks_modules_1_modules_1_modules_conv_pw_modules_1_parameters_weight_
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
        l_self_modules_blocks_modules_1_modules_1_modules_conv_pwl_modules_0_parameters_weight_ = L_self_modules_blocks_modules_1_modules_1_modules_conv_pwl_modules_0_parameters_weight_
        l_self_modules_blocks_modules_1_modules_1_modules_conv_pwl_modules_1_parameters_weight_ = L_self_modules_blocks_modules_1_modules_1_modules_conv_pwl_modules_1_parameters_weight_
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
        l_self_modules_blocks_modules_2_modules_0_modules_conv_dw_modules_0_parameters_weight_ = L_self_modules_blocks_modules_2_modules_0_modules_conv_dw_modules_0_parameters_weight_
        l_self_modules_blocks_modules_2_modules_0_modules_conv_dw_modules_1_parameters_weight_ = L_self_modules_blocks_modules_2_modules_0_modules_conv_dw_modules_1_parameters_weight_
        l_self_modules_blocks_modules_2_modules_0_modules_conv_dw_modules_2_parameters_weight_ = L_self_modules_blocks_modules_2_modules_0_modules_conv_dw_modules_2_parameters_weight_
        l_self_modules_blocks_modules_2_modules_0_modules_conv_dw_modules_3_parameters_weight_ = L_self_modules_blocks_modules_2_modules_0_modules_conv_dw_modules_3_parameters_weight_
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
        l_self_modules_blocks_modules_2_modules_1_modules_conv_pw_modules_0_parameters_weight_ = L_self_modules_blocks_modules_2_modules_1_modules_conv_pw_modules_0_parameters_weight_
        l_self_modules_blocks_modules_2_modules_1_modules_conv_pw_modules_1_parameters_weight_ = L_self_modules_blocks_modules_2_modules_1_modules_conv_pw_modules_1_parameters_weight_
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
        l_self_modules_blocks_modules_2_modules_1_modules_conv_dw_modules_0_parameters_weight_ = L_self_modules_blocks_modules_2_modules_1_modules_conv_dw_modules_0_parameters_weight_
        l_self_modules_blocks_modules_2_modules_1_modules_conv_dw_modules_1_parameters_weight_ = L_self_modules_blocks_modules_2_modules_1_modules_conv_dw_modules_1_parameters_weight_
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
        l_self_modules_blocks_modules_2_modules_1_modules_conv_pwl_modules_0_parameters_weight_ = L_self_modules_blocks_modules_2_modules_1_modules_conv_pwl_modules_0_parameters_weight_
        l_self_modules_blocks_modules_2_modules_1_modules_conv_pwl_modules_1_parameters_weight_ = L_self_modules_blocks_modules_2_modules_1_modules_conv_pwl_modules_1_parameters_weight_
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
        l_self_modules_blocks_modules_2_modules_2_modules_conv_pw_modules_0_parameters_weight_ = L_self_modules_blocks_modules_2_modules_2_modules_conv_pw_modules_0_parameters_weight_
        l_self_modules_blocks_modules_2_modules_2_modules_conv_pw_modules_1_parameters_weight_ = L_self_modules_blocks_modules_2_modules_2_modules_conv_pw_modules_1_parameters_weight_
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
        l_self_modules_blocks_modules_2_modules_2_modules_conv_dw_modules_0_parameters_weight_ = L_self_modules_blocks_modules_2_modules_2_modules_conv_dw_modules_0_parameters_weight_
        l_self_modules_blocks_modules_2_modules_2_modules_conv_dw_modules_1_parameters_weight_ = L_self_modules_blocks_modules_2_modules_2_modules_conv_dw_modules_1_parameters_weight_
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
        l_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_2_modules_2_modules_conv_pwl_modules_0_parameters_weight_ = L_self_modules_blocks_modules_2_modules_2_modules_conv_pwl_modules_0_parameters_weight_
        l_self_modules_blocks_modules_2_modules_2_modules_conv_pwl_modules_1_parameters_weight_ = L_self_modules_blocks_modules_2_modules_2_modules_conv_pwl_modules_1_parameters_weight_
        l_self_modules_blocks_modules_2_modules_2_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_2_modules_2_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_2_modules_2_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_2_modules_2_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_2_modules_2_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_2_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_2_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_2_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_3_modules_conv_pw_modules_0_parameters_weight_ = L_self_modules_blocks_modules_2_modules_3_modules_conv_pw_modules_0_parameters_weight_
        l_self_modules_blocks_modules_2_modules_3_modules_conv_pw_modules_1_parameters_weight_ = L_self_modules_blocks_modules_2_modules_3_modules_conv_pw_modules_1_parameters_weight_
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
        l_self_modules_blocks_modules_2_modules_3_modules_conv_dw_modules_0_parameters_weight_ = L_self_modules_blocks_modules_2_modules_3_modules_conv_dw_modules_0_parameters_weight_
        l_self_modules_blocks_modules_2_modules_3_modules_conv_dw_modules_1_parameters_weight_ = L_self_modules_blocks_modules_2_modules_3_modules_conv_dw_modules_1_parameters_weight_
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
        l_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_2_modules_3_modules_conv_pwl_modules_0_parameters_weight_ = L_self_modules_blocks_modules_2_modules_3_modules_conv_pwl_modules_0_parameters_weight_
        l_self_modules_blocks_modules_2_modules_3_modules_conv_pwl_modules_1_parameters_weight_ = L_self_modules_blocks_modules_2_modules_3_modules_conv_pwl_modules_1_parameters_weight_
        l_self_modules_blocks_modules_2_modules_3_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_2_modules_3_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_2_modules_3_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_2_modules_3_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_2_modules_3_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_3_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_3_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_3_modules_bn3_parameters_bias_
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
        l_self_modules_blocks_modules_3_modules_0_modules_conv_dw_modules_0_parameters_weight_ = L_self_modules_blocks_modules_3_modules_0_modules_conv_dw_modules_0_parameters_weight_
        l_self_modules_blocks_modules_3_modules_0_modules_conv_dw_modules_1_parameters_weight_ = L_self_modules_blocks_modules_3_modules_0_modules_conv_dw_modules_1_parameters_weight_
        l_self_modules_blocks_modules_3_modules_0_modules_conv_dw_modules_2_parameters_weight_ = L_self_modules_blocks_modules_3_modules_0_modules_conv_dw_modules_2_parameters_weight_
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
        l_self_modules_blocks_modules_3_modules_1_modules_conv_pw_modules_0_parameters_weight_ = L_self_modules_blocks_modules_3_modules_1_modules_conv_pw_modules_0_parameters_weight_
        l_self_modules_blocks_modules_3_modules_1_modules_conv_pw_modules_1_parameters_weight_ = L_self_modules_blocks_modules_3_modules_1_modules_conv_pw_modules_1_parameters_weight_
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
        l_self_modules_blocks_modules_3_modules_1_modules_conv_dw_modules_0_parameters_weight_ = L_self_modules_blocks_modules_3_modules_1_modules_conv_dw_modules_0_parameters_weight_
        l_self_modules_blocks_modules_3_modules_1_modules_conv_dw_modules_1_parameters_weight_ = L_self_modules_blocks_modules_3_modules_1_modules_conv_dw_modules_1_parameters_weight_
        l_self_modules_blocks_modules_3_modules_1_modules_conv_dw_modules_2_parameters_weight_ = L_self_modules_blocks_modules_3_modules_1_modules_conv_dw_modules_2_parameters_weight_
        l_self_modules_blocks_modules_3_modules_1_modules_conv_dw_modules_3_parameters_weight_ = L_self_modules_blocks_modules_3_modules_1_modules_conv_dw_modules_3_parameters_weight_
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
        l_self_modules_blocks_modules_3_modules_1_modules_conv_pwl_modules_0_parameters_weight_ = L_self_modules_blocks_modules_3_modules_1_modules_conv_pwl_modules_0_parameters_weight_
        l_self_modules_blocks_modules_3_modules_1_modules_conv_pwl_modules_1_parameters_weight_ = L_self_modules_blocks_modules_3_modules_1_modules_conv_pwl_modules_1_parameters_weight_
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
        l_self_modules_blocks_modules_3_modules_2_modules_conv_pw_modules_0_parameters_weight_ = L_self_modules_blocks_modules_3_modules_2_modules_conv_pw_modules_0_parameters_weight_
        l_self_modules_blocks_modules_3_modules_2_modules_conv_pw_modules_1_parameters_weight_ = L_self_modules_blocks_modules_3_modules_2_modules_conv_pw_modules_1_parameters_weight_
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
        l_self_modules_blocks_modules_3_modules_2_modules_conv_dw_modules_0_parameters_weight_ = L_self_modules_blocks_modules_3_modules_2_modules_conv_dw_modules_0_parameters_weight_
        l_self_modules_blocks_modules_3_modules_2_modules_conv_dw_modules_1_parameters_weight_ = L_self_modules_blocks_modules_3_modules_2_modules_conv_dw_modules_1_parameters_weight_
        l_self_modules_blocks_modules_3_modules_2_modules_conv_dw_modules_2_parameters_weight_ = L_self_modules_blocks_modules_3_modules_2_modules_conv_dw_modules_2_parameters_weight_
        l_self_modules_blocks_modules_3_modules_2_modules_conv_dw_modules_3_parameters_weight_ = L_self_modules_blocks_modules_3_modules_2_modules_conv_dw_modules_3_parameters_weight_
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
        l_self_modules_blocks_modules_3_modules_2_modules_conv_pwl_modules_0_parameters_weight_ = L_self_modules_blocks_modules_3_modules_2_modules_conv_pwl_modules_0_parameters_weight_
        l_self_modules_blocks_modules_3_modules_2_modules_conv_pwl_modules_1_parameters_weight_ = L_self_modules_blocks_modules_3_modules_2_modules_conv_pwl_modules_1_parameters_weight_
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
        l_self_modules_blocks_modules_3_modules_3_modules_conv_pw_modules_0_parameters_weight_ = L_self_modules_blocks_modules_3_modules_3_modules_conv_pw_modules_0_parameters_weight_
        l_self_modules_blocks_modules_3_modules_3_modules_conv_pw_modules_1_parameters_weight_ = L_self_modules_blocks_modules_3_modules_3_modules_conv_pw_modules_1_parameters_weight_
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
        l_self_modules_blocks_modules_3_modules_3_modules_conv_dw_modules_0_parameters_weight_ = L_self_modules_blocks_modules_3_modules_3_modules_conv_dw_modules_0_parameters_weight_
        l_self_modules_blocks_modules_3_modules_3_modules_conv_dw_modules_1_parameters_weight_ = L_self_modules_blocks_modules_3_modules_3_modules_conv_dw_modules_1_parameters_weight_
        l_self_modules_blocks_modules_3_modules_3_modules_conv_dw_modules_2_parameters_weight_ = L_self_modules_blocks_modules_3_modules_3_modules_conv_dw_modules_2_parameters_weight_
        l_self_modules_blocks_modules_3_modules_3_modules_conv_dw_modules_3_parameters_weight_ = L_self_modules_blocks_modules_3_modules_3_modules_conv_dw_modules_3_parameters_weight_
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
        l_self_modules_blocks_modules_3_modules_3_modules_conv_pwl_modules_0_parameters_weight_ = L_self_modules_blocks_modules_3_modules_3_modules_conv_pwl_modules_0_parameters_weight_
        l_self_modules_blocks_modules_3_modules_3_modules_conv_pwl_modules_1_parameters_weight_ = L_self_modules_blocks_modules_3_modules_3_modules_conv_pwl_modules_1_parameters_weight_
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
        l_self_modules_blocks_modules_4_modules_1_modules_conv_pw_modules_0_parameters_weight_ = L_self_modules_blocks_modules_4_modules_1_modules_conv_pw_modules_0_parameters_weight_
        l_self_modules_blocks_modules_4_modules_1_modules_conv_pw_modules_1_parameters_weight_ = L_self_modules_blocks_modules_4_modules_1_modules_conv_pw_modules_1_parameters_weight_
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
        l_self_modules_blocks_modules_4_modules_1_modules_conv_dw_modules_0_parameters_weight_ = L_self_modules_blocks_modules_4_modules_1_modules_conv_dw_modules_0_parameters_weight_
        l_self_modules_blocks_modules_4_modules_1_modules_conv_dw_modules_1_parameters_weight_ = L_self_modules_blocks_modules_4_modules_1_modules_conv_dw_modules_1_parameters_weight_
        l_self_modules_blocks_modules_4_modules_1_modules_conv_dw_modules_2_parameters_weight_ = L_self_modules_blocks_modules_4_modules_1_modules_conv_dw_modules_2_parameters_weight_
        l_self_modules_blocks_modules_4_modules_1_modules_conv_dw_modules_3_parameters_weight_ = L_self_modules_blocks_modules_4_modules_1_modules_conv_dw_modules_3_parameters_weight_
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
        l_self_modules_blocks_modules_4_modules_1_modules_conv_pwl_modules_0_parameters_weight_ = L_self_modules_blocks_modules_4_modules_1_modules_conv_pwl_modules_0_parameters_weight_
        l_self_modules_blocks_modules_4_modules_1_modules_conv_pwl_modules_1_parameters_weight_ = L_self_modules_blocks_modules_4_modules_1_modules_conv_pwl_modules_1_parameters_weight_
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
        l_self_modules_blocks_modules_4_modules_2_modules_conv_pw_modules_0_parameters_weight_ = L_self_modules_blocks_modules_4_modules_2_modules_conv_pw_modules_0_parameters_weight_
        l_self_modules_blocks_modules_4_modules_2_modules_conv_pw_modules_1_parameters_weight_ = L_self_modules_blocks_modules_4_modules_2_modules_conv_pw_modules_1_parameters_weight_
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
        l_self_modules_blocks_modules_4_modules_2_modules_conv_dw_modules_0_parameters_weight_ = L_self_modules_blocks_modules_4_modules_2_modules_conv_dw_modules_0_parameters_weight_
        l_self_modules_blocks_modules_4_modules_2_modules_conv_dw_modules_1_parameters_weight_ = L_self_modules_blocks_modules_4_modules_2_modules_conv_dw_modules_1_parameters_weight_
        l_self_modules_blocks_modules_4_modules_2_modules_conv_dw_modules_2_parameters_weight_ = L_self_modules_blocks_modules_4_modules_2_modules_conv_dw_modules_2_parameters_weight_
        l_self_modules_blocks_modules_4_modules_2_modules_conv_dw_modules_3_parameters_weight_ = L_self_modules_blocks_modules_4_modules_2_modules_conv_dw_modules_3_parameters_weight_
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
        l_self_modules_blocks_modules_4_modules_2_modules_conv_pwl_modules_0_parameters_weight_ = L_self_modules_blocks_modules_4_modules_2_modules_conv_pwl_modules_0_parameters_weight_
        l_self_modules_blocks_modules_4_modules_2_modules_conv_pwl_modules_1_parameters_weight_ = L_self_modules_blocks_modules_4_modules_2_modules_conv_pwl_modules_1_parameters_weight_
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
        l_self_modules_blocks_modules_4_modules_3_modules_conv_pw_modules_0_parameters_weight_ = L_self_modules_blocks_modules_4_modules_3_modules_conv_pw_modules_0_parameters_weight_
        l_self_modules_blocks_modules_4_modules_3_modules_conv_pw_modules_1_parameters_weight_ = L_self_modules_blocks_modules_4_modules_3_modules_conv_pw_modules_1_parameters_weight_
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
        l_self_modules_blocks_modules_4_modules_3_modules_conv_dw_modules_0_parameters_weight_ = L_self_modules_blocks_modules_4_modules_3_modules_conv_dw_modules_0_parameters_weight_
        l_self_modules_blocks_modules_4_modules_3_modules_conv_dw_modules_1_parameters_weight_ = L_self_modules_blocks_modules_4_modules_3_modules_conv_dw_modules_1_parameters_weight_
        l_self_modules_blocks_modules_4_modules_3_modules_conv_dw_modules_2_parameters_weight_ = L_self_modules_blocks_modules_4_modules_3_modules_conv_dw_modules_2_parameters_weight_
        l_self_modules_blocks_modules_4_modules_3_modules_conv_dw_modules_3_parameters_weight_ = L_self_modules_blocks_modules_4_modules_3_modules_conv_dw_modules_3_parameters_weight_
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
        l_self_modules_blocks_modules_4_modules_3_modules_conv_pwl_modules_0_parameters_weight_ = L_self_modules_blocks_modules_4_modules_3_modules_conv_pwl_modules_0_parameters_weight_
        l_self_modules_blocks_modules_4_modules_3_modules_conv_pwl_modules_1_parameters_weight_ = L_self_modules_blocks_modules_4_modules_3_modules_conv_pwl_modules_1_parameters_weight_
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
        l_self_modules_blocks_modules_5_modules_0_modules_conv_dw_modules_0_parameters_weight_ = L_self_modules_blocks_modules_5_modules_0_modules_conv_dw_modules_0_parameters_weight_
        l_self_modules_blocks_modules_5_modules_0_modules_conv_dw_modules_1_parameters_weight_ = L_self_modules_blocks_modules_5_modules_0_modules_conv_dw_modules_1_parameters_weight_
        l_self_modules_blocks_modules_5_modules_0_modules_conv_dw_modules_2_parameters_weight_ = L_self_modules_blocks_modules_5_modules_0_modules_conv_dw_modules_2_parameters_weight_
        l_self_modules_blocks_modules_5_modules_0_modules_conv_dw_modules_3_parameters_weight_ = L_self_modules_blocks_modules_5_modules_0_modules_conv_dw_modules_3_parameters_weight_
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
        l_self_modules_blocks_modules_5_modules_1_modules_conv_dw_modules_0_parameters_weight_ = L_self_modules_blocks_modules_5_modules_1_modules_conv_dw_modules_0_parameters_weight_
        l_self_modules_blocks_modules_5_modules_1_modules_conv_dw_modules_1_parameters_weight_ = L_self_modules_blocks_modules_5_modules_1_modules_conv_dw_modules_1_parameters_weight_
        l_self_modules_blocks_modules_5_modules_1_modules_conv_dw_modules_2_parameters_weight_ = L_self_modules_blocks_modules_5_modules_1_modules_conv_dw_modules_2_parameters_weight_
        l_self_modules_blocks_modules_5_modules_1_modules_conv_dw_modules_3_parameters_weight_ = L_self_modules_blocks_modules_5_modules_1_modules_conv_dw_modules_3_parameters_weight_
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
        l_self_modules_blocks_modules_5_modules_1_modules_conv_pwl_modules_0_parameters_weight_ = L_self_modules_blocks_modules_5_modules_1_modules_conv_pwl_modules_0_parameters_weight_
        l_self_modules_blocks_modules_5_modules_1_modules_conv_pwl_modules_1_parameters_weight_ = L_self_modules_blocks_modules_5_modules_1_modules_conv_pwl_modules_1_parameters_weight_
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
        l_self_modules_blocks_modules_5_modules_2_modules_conv_dw_modules_0_parameters_weight_ = L_self_modules_blocks_modules_5_modules_2_modules_conv_dw_modules_0_parameters_weight_
        l_self_modules_blocks_modules_5_modules_2_modules_conv_dw_modules_1_parameters_weight_ = L_self_modules_blocks_modules_5_modules_2_modules_conv_dw_modules_1_parameters_weight_
        l_self_modules_blocks_modules_5_modules_2_modules_conv_dw_modules_2_parameters_weight_ = L_self_modules_blocks_modules_5_modules_2_modules_conv_dw_modules_2_parameters_weight_
        l_self_modules_blocks_modules_5_modules_2_modules_conv_dw_modules_3_parameters_weight_ = L_self_modules_blocks_modules_5_modules_2_modules_conv_dw_modules_3_parameters_weight_
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
        l_self_modules_blocks_modules_5_modules_2_modules_conv_pwl_modules_0_parameters_weight_ = L_self_modules_blocks_modules_5_modules_2_modules_conv_pwl_modules_0_parameters_weight_
        l_self_modules_blocks_modules_5_modules_2_modules_conv_pwl_modules_1_parameters_weight_ = L_self_modules_blocks_modules_5_modules_2_modules_conv_pwl_modules_1_parameters_weight_
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
        l_self_modules_blocks_modules_5_modules_3_modules_conv_dw_modules_0_parameters_weight_ = L_self_modules_blocks_modules_5_modules_3_modules_conv_dw_modules_0_parameters_weight_
        l_self_modules_blocks_modules_5_modules_3_modules_conv_dw_modules_1_parameters_weight_ = L_self_modules_blocks_modules_5_modules_3_modules_conv_dw_modules_1_parameters_weight_
        l_self_modules_blocks_modules_5_modules_3_modules_conv_dw_modules_2_parameters_weight_ = L_self_modules_blocks_modules_5_modules_3_modules_conv_dw_modules_2_parameters_weight_
        l_self_modules_blocks_modules_5_modules_3_modules_conv_dw_modules_3_parameters_weight_ = L_self_modules_blocks_modules_5_modules_3_modules_conv_dw_modules_3_parameters_weight_
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
        l_self_modules_blocks_modules_5_modules_3_modules_conv_pwl_modules_0_parameters_weight_ = L_self_modules_blocks_modules_5_modules_3_modules_conv_pwl_modules_0_parameters_weight_
        l_self_modules_blocks_modules_5_modules_3_modules_conv_pwl_modules_1_parameters_weight_ = L_self_modules_blocks_modules_5_modules_3_modules_conv_pwl_modules_1_parameters_weight_
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
        x_2 = torch.nn.functional.relu(x_1, inplace=True)
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
        l_self_modules_blocks_modules_0_modules_0_modules_conv_dw_parameters_weight_ = (
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
        x_5 = torch.nn.functional.relu(x_4, inplace=True)
        x_4 = None
        x_6 = torch.conv2d(
            x_5,
            l_self_modules_blocks_modules_0_modules_0_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_5 = (
            l_self_modules_blocks_modules_0_modules_0_modules_conv_pw_parameters_weight_
        ) = None
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
        split = torch.functional.split(x_8, [16, 16], 1)
        x_8 = None
        getitem = split[0]
        getitem_1 = split[1]
        split = None
        conv2d_3 = torch.conv2d(
            getitem,
            l_self_modules_blocks_modules_1_modules_0_modules_conv_pw_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        getitem = l_self_modules_blocks_modules_1_modules_0_modules_conv_pw_modules_0_parameters_weight_ = (None)
        conv2d_4 = torch.conv2d(
            getitem_1,
            l_self_modules_blocks_modules_1_modules_0_modules_conv_pw_modules_1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        getitem_1 = l_self_modules_blocks_modules_1_modules_0_modules_conv_pw_modules_1_parameters_weight_ = (None)
        x_9 = torch.cat([conv2d_3, conv2d_4], 1)
        conv2d_3 = conv2d_4 = None
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
        x_11 = torch.nn.functional.relu(x_10, inplace=True)
        x_10 = None
        split_1 = torch.functional.split(x_11, [64, 64, 64], 1)
        x_11 = None
        getitem_2 = split_1[0]
        getitem_3 = split_1[1]
        getitem_4 = split_1[2]
        split_1 = None
        conv2d_5 = torch.conv2d(
            getitem_2,
            l_self_modules_blocks_modules_1_modules_0_modules_conv_dw_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            64,
        )
        getitem_2 = l_self_modules_blocks_modules_1_modules_0_modules_conv_dw_modules_0_parameters_weight_ = (None)
        conv2d_6 = torch.conv2d(
            getitem_3,
            l_self_modules_blocks_modules_1_modules_0_modules_conv_dw_modules_1_parameters_weight_,
            None,
            (2, 2),
            (2, 2),
            (1, 1),
            64,
        )
        getitem_3 = l_self_modules_blocks_modules_1_modules_0_modules_conv_dw_modules_1_parameters_weight_ = (None)
        conv2d_7 = torch.conv2d(
            getitem_4,
            l_self_modules_blocks_modules_1_modules_0_modules_conv_dw_modules_2_parameters_weight_,
            None,
            (2, 2),
            (3, 3),
            (1, 1),
            64,
        )
        getitem_4 = l_self_modules_blocks_modules_1_modules_0_modules_conv_dw_modules_2_parameters_weight_ = (None)
        x_12 = torch.cat([conv2d_5, conv2d_6, conv2d_7], 1)
        conv2d_5 = conv2d_6 = conv2d_7 = None
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
        x_14 = torch.nn.functional.relu(x_13, inplace=True)
        x_13 = None
        split_2 = torch.functional.split(x_14, [96, 96], 1)
        x_14 = None
        getitem_5 = split_2[0]
        getitem_6 = split_2[1]
        split_2 = None
        conv2d_8 = torch.conv2d(
            getitem_5,
            l_self_modules_blocks_modules_1_modules_0_modules_conv_pwl_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        getitem_5 = l_self_modules_blocks_modules_1_modules_0_modules_conv_pwl_modules_0_parameters_weight_ = (None)
        conv2d_9 = torch.conv2d(
            getitem_6,
            l_self_modules_blocks_modules_1_modules_0_modules_conv_pwl_modules_1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        getitem_6 = l_self_modules_blocks_modules_1_modules_0_modules_conv_pwl_modules_1_parameters_weight_ = (None)
        x_15 = torch.cat([conv2d_8, conv2d_9], 1)
        conv2d_8 = conv2d_9 = None
        x_16 = torch.nn.functional.batch_norm(
            x_15,
            l_self_modules_blocks_modules_1_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_15 = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn3_parameters_bias_
        ) = None
        split_3 = torch.functional.split(x_16, [20, 20], 1)
        getitem_7 = split_3[0]
        getitem_8 = split_3[1]
        split_3 = None
        conv2d_10 = torch.conv2d(
            getitem_7,
            l_self_modules_blocks_modules_1_modules_1_modules_conv_pw_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        getitem_7 = l_self_modules_blocks_modules_1_modules_1_modules_conv_pw_modules_0_parameters_weight_ = (None)
        conv2d_11 = torch.conv2d(
            getitem_8,
            l_self_modules_blocks_modules_1_modules_1_modules_conv_pw_modules_1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        getitem_8 = l_self_modules_blocks_modules_1_modules_1_modules_conv_pw_modules_1_parameters_weight_ = (None)
        x_17 = torch.cat([conv2d_10, conv2d_11], 1)
        conv2d_10 = conv2d_11 = None
        x_18 = torch.nn.functional.batch_norm(
            x_17,
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_17 = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_bias_
        ) = None
        x_19 = torch.nn.functional.relu(x_18, inplace=True)
        x_18 = None
        x_20 = torch.conv2d(
            x_19,
            l_self_modules_blocks_modules_1_modules_1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            120,
        )
        x_19 = (
            l_self_modules_blocks_modules_1_modules_1_modules_conv_dw_parameters_weight_
        ) = None
        x_21 = torch.nn.functional.batch_norm(
            x_20,
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_20 = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_bias_
        ) = None
        x_22 = torch.nn.functional.relu(x_21, inplace=True)
        x_21 = None
        split_4 = torch.functional.split(x_22, [60, 60], 1)
        x_22 = None
        getitem_9 = split_4[0]
        getitem_10 = split_4[1]
        split_4 = None
        conv2d_13 = torch.conv2d(
            getitem_9,
            l_self_modules_blocks_modules_1_modules_1_modules_conv_pwl_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        getitem_9 = l_self_modules_blocks_modules_1_modules_1_modules_conv_pwl_modules_0_parameters_weight_ = (None)
        conv2d_14 = torch.conv2d(
            getitem_10,
            l_self_modules_blocks_modules_1_modules_1_modules_conv_pwl_modules_1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        getitem_10 = l_self_modules_blocks_modules_1_modules_1_modules_conv_pwl_modules_1_parameters_weight_ = (None)
        x_23 = torch.cat([conv2d_13, conv2d_14], 1)
        conv2d_13 = conv2d_14 = None
        x_24 = torch.nn.functional.batch_norm(
            x_23,
            l_self_modules_blocks_modules_1_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_23 = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn3_parameters_bias_
        ) = None
        x_25 = x_24 + x_16
        x_24 = x_16 = None
        x_26 = torch.conv2d(
            x_25,
            l_self_modules_blocks_modules_2_modules_0_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_25 = (
            l_self_modules_blocks_modules_2_modules_0_modules_conv_pw_parameters_weight_
        ) = None
        x_27 = torch.nn.functional.batch_norm(
            x_26,
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_26 = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_28 = torch.nn.functional.silu(x_27, inplace=True)
        x_27 = None
        split_5 = torch.functional.split(x_28, [60, 60, 60, 60], 1)
        x_28 = None
        getitem_11 = split_5[0]
        getitem_12 = split_5[1]
        getitem_13 = split_5[2]
        getitem_14 = split_5[3]
        split_5 = None
        conv2d_16 = torch.conv2d(
            getitem_11,
            l_self_modules_blocks_modules_2_modules_0_modules_conv_dw_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            60,
        )
        getitem_11 = l_self_modules_blocks_modules_2_modules_0_modules_conv_dw_modules_0_parameters_weight_ = (None)
        conv2d_17 = torch.conv2d(
            getitem_12,
            l_self_modules_blocks_modules_2_modules_0_modules_conv_dw_modules_1_parameters_weight_,
            None,
            (2, 2),
            (2, 2),
            (1, 1),
            60,
        )
        getitem_12 = l_self_modules_blocks_modules_2_modules_0_modules_conv_dw_modules_1_parameters_weight_ = (None)
        conv2d_18 = torch.conv2d(
            getitem_13,
            l_self_modules_blocks_modules_2_modules_0_modules_conv_dw_modules_2_parameters_weight_,
            None,
            (2, 2),
            (3, 3),
            (1, 1),
            60,
        )
        getitem_13 = l_self_modules_blocks_modules_2_modules_0_modules_conv_dw_modules_2_parameters_weight_ = (None)
        conv2d_19 = torch.conv2d(
            getitem_14,
            l_self_modules_blocks_modules_2_modules_0_modules_conv_dw_modules_3_parameters_weight_,
            None,
            (2, 2),
            (4, 4),
            (1, 1),
            60,
        )
        getitem_14 = l_self_modules_blocks_modules_2_modules_0_modules_conv_dw_modules_3_parameters_weight_ = (None)
        x_29 = torch.cat([conv2d_16, conv2d_17, conv2d_18, conv2d_19], 1)
        conv2d_16 = conv2d_17 = conv2d_18 = conv2d_19 = None
        x_30 = torch.nn.functional.batch_norm(
            x_29,
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_29 = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_bias_
        ) = None
        x_31 = torch.nn.functional.silu(x_30, inplace=True)
        x_30 = None
        x_se = x_31.mean((2, 3), keepdim=True)
        x_se_1 = torch.conv2d(
            x_se,
            l_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se = l_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_2 = torch.nn.functional.silu(x_se_1, inplace=True)
        x_se_1 = None
        x_se_3 = torch.conv2d(
            x_se_2,
            l_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_2 = l_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid = torch.sigmoid(x_se_3)
        x_se_3 = None
        x_32 = x_31 * sigmoid
        x_31 = sigmoid = None
        x_33 = torch.conv2d(
            x_32,
            l_self_modules_blocks_modules_2_modules_0_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_32 = l_self_modules_blocks_modules_2_modules_0_modules_conv_pwl_parameters_weight_ = (None)
        x_34 = torch.nn.functional.batch_norm(
            x_33,
            l_self_modules_blocks_modules_2_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_33 = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn3_parameters_bias_
        ) = None
        split_6 = torch.functional.split(x_34, [28, 28], 1)
        getitem_15 = split_6[0]
        getitem_16 = split_6[1]
        split_6 = None
        conv2d_23 = torch.conv2d(
            getitem_15,
            l_self_modules_blocks_modules_2_modules_1_modules_conv_pw_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        getitem_15 = l_self_modules_blocks_modules_2_modules_1_modules_conv_pw_modules_0_parameters_weight_ = (None)
        conv2d_24 = torch.conv2d(
            getitem_16,
            l_self_modules_blocks_modules_2_modules_1_modules_conv_pw_modules_1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        getitem_16 = l_self_modules_blocks_modules_2_modules_1_modules_conv_pw_modules_1_parameters_weight_ = (None)
        x_35 = torch.cat([conv2d_23, conv2d_24], 1)
        conv2d_23 = conv2d_24 = None
        x_36 = torch.nn.functional.batch_norm(
            x_35,
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_35 = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_bias_
        ) = None
        x_37 = torch.nn.functional.silu(x_36, inplace=True)
        x_36 = None
        split_7 = torch.functional.split(x_37, [168, 168], 1)
        x_37 = None
        getitem_17 = split_7[0]
        getitem_18 = split_7[1]
        split_7 = None
        conv2d_25 = torch.conv2d(
            getitem_17,
            l_self_modules_blocks_modules_2_modules_1_modules_conv_dw_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            168,
        )
        getitem_17 = l_self_modules_blocks_modules_2_modules_1_modules_conv_dw_modules_0_parameters_weight_ = (None)
        conv2d_26 = torch.conv2d(
            getitem_18,
            l_self_modules_blocks_modules_2_modules_1_modules_conv_dw_modules_1_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            168,
        )
        getitem_18 = l_self_modules_blocks_modules_2_modules_1_modules_conv_dw_modules_1_parameters_weight_ = (None)
        x_38 = torch.cat([conv2d_25, conv2d_26], 1)
        conv2d_25 = conv2d_26 = None
        x_39 = torch.nn.functional.batch_norm(
            x_38,
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_38 = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_bias_
        ) = None
        x_40 = torch.nn.functional.silu(x_39, inplace=True)
        x_39 = None
        x_se_4 = x_40.mean((2, 3), keepdim=True)
        x_se_5 = torch.conv2d(
            x_se_4,
            l_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_4 = l_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_6 = torch.nn.functional.silu(x_se_5, inplace=True)
        x_se_5 = None
        x_se_7 = torch.conv2d(
            x_se_6,
            l_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_6 = l_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_1 = torch.sigmoid(x_se_7)
        x_se_7 = None
        x_41 = x_40 * sigmoid_1
        x_40 = sigmoid_1 = None
        split_8 = torch.functional.split(x_41, [168, 168], 1)
        x_41 = None
        getitem_19 = split_8[0]
        getitem_20 = split_8[1]
        split_8 = None
        conv2d_29 = torch.conv2d(
            getitem_19,
            l_self_modules_blocks_modules_2_modules_1_modules_conv_pwl_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        getitem_19 = l_self_modules_blocks_modules_2_modules_1_modules_conv_pwl_modules_0_parameters_weight_ = (None)
        conv2d_30 = torch.conv2d(
            getitem_20,
            l_self_modules_blocks_modules_2_modules_1_modules_conv_pwl_modules_1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        getitem_20 = l_self_modules_blocks_modules_2_modules_1_modules_conv_pwl_modules_1_parameters_weight_ = (None)
        x_42 = torch.cat([conv2d_29, conv2d_30], 1)
        conv2d_29 = conv2d_30 = None
        x_43 = torch.nn.functional.batch_norm(
            x_42,
            l_self_modules_blocks_modules_2_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_42 = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn3_parameters_bias_
        ) = None
        x_44 = x_43 + x_34
        x_43 = x_34 = None
        split_9 = torch.functional.split(x_44, [28, 28], 1)
        getitem_21 = split_9[0]
        getitem_22 = split_9[1]
        split_9 = None
        conv2d_31 = torch.conv2d(
            getitem_21,
            l_self_modules_blocks_modules_2_modules_2_modules_conv_pw_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        getitem_21 = l_self_modules_blocks_modules_2_modules_2_modules_conv_pw_modules_0_parameters_weight_ = (None)
        conv2d_32 = torch.conv2d(
            getitem_22,
            l_self_modules_blocks_modules_2_modules_2_modules_conv_pw_modules_1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        getitem_22 = l_self_modules_blocks_modules_2_modules_2_modules_conv_pw_modules_1_parameters_weight_ = (None)
        x_45 = torch.cat([conv2d_31, conv2d_32], 1)
        conv2d_31 = conv2d_32 = None
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
        split_10 = torch.functional.split(x_47, [168, 168], 1)
        x_47 = None
        getitem_23 = split_10[0]
        getitem_24 = split_10[1]
        split_10 = None
        conv2d_33 = torch.conv2d(
            getitem_23,
            l_self_modules_blocks_modules_2_modules_2_modules_conv_dw_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            168,
        )
        getitem_23 = l_self_modules_blocks_modules_2_modules_2_modules_conv_dw_modules_0_parameters_weight_ = (None)
        conv2d_34 = torch.conv2d(
            getitem_24,
            l_self_modules_blocks_modules_2_modules_2_modules_conv_dw_modules_1_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            168,
        )
        getitem_24 = l_self_modules_blocks_modules_2_modules_2_modules_conv_dw_modules_1_parameters_weight_ = (None)
        x_48 = torch.cat([conv2d_33, conv2d_34], 1)
        conv2d_33 = conv2d_34 = None
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
        x_50 = torch.nn.functional.silu(x_49, inplace=True)
        x_49 = None
        x_se_8 = x_50.mean((2, 3), keepdim=True)
        x_se_9 = torch.conv2d(
            x_se_8,
            l_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_8 = l_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_10 = torch.nn.functional.silu(x_se_9, inplace=True)
        x_se_9 = None
        x_se_11 = torch.conv2d(
            x_se_10,
            l_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_10 = l_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_2 = torch.sigmoid(x_se_11)
        x_se_11 = None
        x_51 = x_50 * sigmoid_2
        x_50 = sigmoid_2 = None
        split_11 = torch.functional.split(x_51, [168, 168], 1)
        x_51 = None
        getitem_25 = split_11[0]
        getitem_26 = split_11[1]
        split_11 = None
        conv2d_37 = torch.conv2d(
            getitem_25,
            l_self_modules_blocks_modules_2_modules_2_modules_conv_pwl_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        getitem_25 = l_self_modules_blocks_modules_2_modules_2_modules_conv_pwl_modules_0_parameters_weight_ = (None)
        conv2d_38 = torch.conv2d(
            getitem_26,
            l_self_modules_blocks_modules_2_modules_2_modules_conv_pwl_modules_1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        getitem_26 = l_self_modules_blocks_modules_2_modules_2_modules_conv_pwl_modules_1_parameters_weight_ = (None)
        x_52 = torch.cat([conv2d_37, conv2d_38], 1)
        conv2d_37 = conv2d_38 = None
        x_53 = torch.nn.functional.batch_norm(
            x_52,
            l_self_modules_blocks_modules_2_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_52 = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn3_parameters_bias_
        ) = None
        x_54 = x_53 + x_44
        x_53 = x_44 = None
        split_12 = torch.functional.split(x_54, [28, 28], 1)
        getitem_27 = split_12[0]
        getitem_28 = split_12[1]
        split_12 = None
        conv2d_39 = torch.conv2d(
            getitem_27,
            l_self_modules_blocks_modules_2_modules_3_modules_conv_pw_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        getitem_27 = l_self_modules_blocks_modules_2_modules_3_modules_conv_pw_modules_0_parameters_weight_ = (None)
        conv2d_40 = torch.conv2d(
            getitem_28,
            l_self_modules_blocks_modules_2_modules_3_modules_conv_pw_modules_1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        getitem_28 = l_self_modules_blocks_modules_2_modules_3_modules_conv_pw_modules_1_parameters_weight_ = (None)
        x_55 = torch.cat([conv2d_39, conv2d_40], 1)
        conv2d_39 = conv2d_40 = None
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
        split_13 = torch.functional.split(x_57, [168, 168], 1)
        x_57 = None
        getitem_29 = split_13[0]
        getitem_30 = split_13[1]
        split_13 = None
        conv2d_41 = torch.conv2d(
            getitem_29,
            l_self_modules_blocks_modules_2_modules_3_modules_conv_dw_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            168,
        )
        getitem_29 = l_self_modules_blocks_modules_2_modules_3_modules_conv_dw_modules_0_parameters_weight_ = (None)
        conv2d_42 = torch.conv2d(
            getitem_30,
            l_self_modules_blocks_modules_2_modules_3_modules_conv_dw_modules_1_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            168,
        )
        getitem_30 = l_self_modules_blocks_modules_2_modules_3_modules_conv_dw_modules_1_parameters_weight_ = (None)
        x_58 = torch.cat([conv2d_41, conv2d_42], 1)
        conv2d_41 = conv2d_42 = None
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
        x_60 = torch.nn.functional.silu(x_59, inplace=True)
        x_59 = None
        x_se_12 = x_60.mean((2, 3), keepdim=True)
        x_se_13 = torch.conv2d(
            x_se_12,
            l_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_12 = l_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_14 = torch.nn.functional.silu(x_se_13, inplace=True)
        x_se_13 = None
        x_se_15 = torch.conv2d(
            x_se_14,
            l_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_14 = l_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_3 = torch.sigmoid(x_se_15)
        x_se_15 = None
        x_61 = x_60 * sigmoid_3
        x_60 = sigmoid_3 = None
        split_14 = torch.functional.split(x_61, [168, 168], 1)
        x_61 = None
        getitem_31 = split_14[0]
        getitem_32 = split_14[1]
        split_14 = None
        conv2d_45 = torch.conv2d(
            getitem_31,
            l_self_modules_blocks_modules_2_modules_3_modules_conv_pwl_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        getitem_31 = l_self_modules_blocks_modules_2_modules_3_modules_conv_pwl_modules_0_parameters_weight_ = (None)
        conv2d_46 = torch.conv2d(
            getitem_32,
            l_self_modules_blocks_modules_2_modules_3_modules_conv_pwl_modules_1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        getitem_32 = l_self_modules_blocks_modules_2_modules_3_modules_conv_pwl_modules_1_parameters_weight_ = (None)
        x_62 = torch.cat([conv2d_45, conv2d_46], 1)
        conv2d_45 = conv2d_46 = None
        x_63 = torch.nn.functional.batch_norm(
            x_62,
            l_self_modules_blocks_modules_2_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_62 = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn3_parameters_bias_
        ) = None
        x_64 = x_63 + x_54
        x_63 = x_54 = None
        x_65 = torch.conv2d(
            x_64,
            l_self_modules_blocks_modules_3_modules_0_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_64 = (
            l_self_modules_blocks_modules_3_modules_0_modules_conv_pw_parameters_weight_
        ) = None
        x_66 = torch.nn.functional.batch_norm(
            x_65,
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_65 = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_67 = torch.nn.functional.silu(x_66, inplace=True)
        x_66 = None
        split_15 = torch.functional.split(x_67, [112, 112, 112], 1)
        x_67 = None
        getitem_33 = split_15[0]
        getitem_34 = split_15[1]
        getitem_35 = split_15[2]
        split_15 = None
        conv2d_48 = torch.conv2d(
            getitem_33,
            l_self_modules_blocks_modules_3_modules_0_modules_conv_dw_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            112,
        )
        getitem_33 = l_self_modules_blocks_modules_3_modules_0_modules_conv_dw_modules_0_parameters_weight_ = (None)
        conv2d_49 = torch.conv2d(
            getitem_34,
            l_self_modules_blocks_modules_3_modules_0_modules_conv_dw_modules_1_parameters_weight_,
            None,
            (2, 2),
            (2, 2),
            (1, 1),
            112,
        )
        getitem_34 = l_self_modules_blocks_modules_3_modules_0_modules_conv_dw_modules_1_parameters_weight_ = (None)
        conv2d_50 = torch.conv2d(
            getitem_35,
            l_self_modules_blocks_modules_3_modules_0_modules_conv_dw_modules_2_parameters_weight_,
            None,
            (2, 2),
            (3, 3),
            (1, 1),
            112,
        )
        getitem_35 = l_self_modules_blocks_modules_3_modules_0_modules_conv_dw_modules_2_parameters_weight_ = (None)
        x_68 = torch.cat([conv2d_48, conv2d_49, conv2d_50], 1)
        conv2d_48 = conv2d_49 = conv2d_50 = None
        x_69 = torch.nn.functional.batch_norm(
            x_68,
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_68 = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_bias_
        ) = None
        x_70 = torch.nn.functional.silu(x_69, inplace=True)
        x_69 = None
        x_se_16 = x_70.mean((2, 3), keepdim=True)
        x_se_17 = torch.conv2d(
            x_se_16,
            l_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_16 = l_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_18 = torch.nn.functional.silu(x_se_17, inplace=True)
        x_se_17 = None
        x_se_19 = torch.conv2d(
            x_se_18,
            l_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_18 = l_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_4 = torch.sigmoid(x_se_19)
        x_se_19 = None
        x_71 = x_70 * sigmoid_4
        x_70 = sigmoid_4 = None
        x_72 = torch.conv2d(
            x_71,
            l_self_modules_blocks_modules_3_modules_0_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_71 = l_self_modules_blocks_modules_3_modules_0_modules_conv_pwl_parameters_weight_ = (None)
        x_73 = torch.nn.functional.batch_norm(
            x_72,
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_72 = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_bias_
        ) = None
        split_16 = torch.functional.split(x_73, [52, 52], 1)
        getitem_36 = split_16[0]
        getitem_37 = split_16[1]
        split_16 = None
        conv2d_54 = torch.conv2d(
            getitem_36,
            l_self_modules_blocks_modules_3_modules_1_modules_conv_pw_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        getitem_36 = l_self_modules_blocks_modules_3_modules_1_modules_conv_pw_modules_0_parameters_weight_ = (None)
        conv2d_55 = torch.conv2d(
            getitem_37,
            l_self_modules_blocks_modules_3_modules_1_modules_conv_pw_modules_1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        getitem_37 = l_self_modules_blocks_modules_3_modules_1_modules_conv_pw_modules_1_parameters_weight_ = (None)
        x_74 = torch.cat([conv2d_54, conv2d_55], 1)
        conv2d_54 = conv2d_55 = None
        x_75 = torch.nn.functional.batch_norm(
            x_74,
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_74 = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_bias_
        ) = None
        x_76 = torch.nn.functional.silu(x_75, inplace=True)
        x_75 = None
        split_17 = torch.functional.split(x_76, [156, 156, 156, 156], 1)
        x_76 = None
        getitem_38 = split_17[0]
        getitem_39 = split_17[1]
        getitem_40 = split_17[2]
        getitem_41 = split_17[3]
        split_17 = None
        conv2d_56 = torch.conv2d(
            getitem_38,
            l_self_modules_blocks_modules_3_modules_1_modules_conv_dw_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            156,
        )
        getitem_38 = l_self_modules_blocks_modules_3_modules_1_modules_conv_dw_modules_0_parameters_weight_ = (None)
        conv2d_57 = torch.conv2d(
            getitem_39,
            l_self_modules_blocks_modules_3_modules_1_modules_conv_dw_modules_1_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            156,
        )
        getitem_39 = l_self_modules_blocks_modules_3_modules_1_modules_conv_dw_modules_1_parameters_weight_ = (None)
        conv2d_58 = torch.conv2d(
            getitem_40,
            l_self_modules_blocks_modules_3_modules_1_modules_conv_dw_modules_2_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            156,
        )
        getitem_40 = l_self_modules_blocks_modules_3_modules_1_modules_conv_dw_modules_2_parameters_weight_ = (None)
        conv2d_59 = torch.conv2d(
            getitem_41,
            l_self_modules_blocks_modules_3_modules_1_modules_conv_dw_modules_3_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (1, 1),
            156,
        )
        getitem_41 = l_self_modules_blocks_modules_3_modules_1_modules_conv_dw_modules_3_parameters_weight_ = (None)
        x_77 = torch.cat([conv2d_56, conv2d_57, conv2d_58, conv2d_59], 1)
        conv2d_56 = conv2d_57 = conv2d_58 = conv2d_59 = None
        x_78 = torch.nn.functional.batch_norm(
            x_77,
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_77 = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_bias_
        ) = None
        x_79 = torch.nn.functional.silu(x_78, inplace=True)
        x_78 = None
        x_se_20 = x_79.mean((2, 3), keepdim=True)
        x_se_21 = torch.conv2d(
            x_se_20,
            l_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_20 = l_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_22 = torch.nn.functional.silu(x_se_21, inplace=True)
        x_se_21 = None
        x_se_23 = torch.conv2d(
            x_se_22,
            l_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_22 = l_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_5 = torch.sigmoid(x_se_23)
        x_se_23 = None
        x_80 = x_79 * sigmoid_5
        x_79 = sigmoid_5 = None
        split_18 = torch.functional.split(x_80, [312, 312], 1)
        x_80 = None
        getitem_42 = split_18[0]
        getitem_43 = split_18[1]
        split_18 = None
        conv2d_62 = torch.conv2d(
            getitem_42,
            l_self_modules_blocks_modules_3_modules_1_modules_conv_pwl_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        getitem_42 = l_self_modules_blocks_modules_3_modules_1_modules_conv_pwl_modules_0_parameters_weight_ = (None)
        conv2d_63 = torch.conv2d(
            getitem_43,
            l_self_modules_blocks_modules_3_modules_1_modules_conv_pwl_modules_1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        getitem_43 = l_self_modules_blocks_modules_3_modules_1_modules_conv_pwl_modules_1_parameters_weight_ = (None)
        x_81 = torch.cat([conv2d_62, conv2d_63], 1)
        conv2d_62 = conv2d_63 = None
        x_82 = torch.nn.functional.batch_norm(
            x_81,
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_81 = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_bias_
        ) = None
        x_83 = x_82 + x_73
        x_82 = x_73 = None
        split_19 = torch.functional.split(x_83, [52, 52], 1)
        getitem_44 = split_19[0]
        getitem_45 = split_19[1]
        split_19 = None
        conv2d_64 = torch.conv2d(
            getitem_44,
            l_self_modules_blocks_modules_3_modules_2_modules_conv_pw_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        getitem_44 = l_self_modules_blocks_modules_3_modules_2_modules_conv_pw_modules_0_parameters_weight_ = (None)
        conv2d_65 = torch.conv2d(
            getitem_45,
            l_self_modules_blocks_modules_3_modules_2_modules_conv_pw_modules_1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        getitem_45 = l_self_modules_blocks_modules_3_modules_2_modules_conv_pw_modules_1_parameters_weight_ = (None)
        x_84 = torch.cat([conv2d_64, conv2d_65], 1)
        conv2d_64 = conv2d_65 = None
        x_85 = torch.nn.functional.batch_norm(
            x_84,
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_84 = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_bias_
        ) = None
        x_86 = torch.nn.functional.silu(x_85, inplace=True)
        x_85 = None
        split_20 = torch.functional.split(x_86, [156, 156, 156, 156], 1)
        x_86 = None
        getitem_46 = split_20[0]
        getitem_47 = split_20[1]
        getitem_48 = split_20[2]
        getitem_49 = split_20[3]
        split_20 = None
        conv2d_66 = torch.conv2d(
            getitem_46,
            l_self_modules_blocks_modules_3_modules_2_modules_conv_dw_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            156,
        )
        getitem_46 = l_self_modules_blocks_modules_3_modules_2_modules_conv_dw_modules_0_parameters_weight_ = (None)
        conv2d_67 = torch.conv2d(
            getitem_47,
            l_self_modules_blocks_modules_3_modules_2_modules_conv_dw_modules_1_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            156,
        )
        getitem_47 = l_self_modules_blocks_modules_3_modules_2_modules_conv_dw_modules_1_parameters_weight_ = (None)
        conv2d_68 = torch.conv2d(
            getitem_48,
            l_self_modules_blocks_modules_3_modules_2_modules_conv_dw_modules_2_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            156,
        )
        getitem_48 = l_self_modules_blocks_modules_3_modules_2_modules_conv_dw_modules_2_parameters_weight_ = (None)
        conv2d_69 = torch.conv2d(
            getitem_49,
            l_self_modules_blocks_modules_3_modules_2_modules_conv_dw_modules_3_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (1, 1),
            156,
        )
        getitem_49 = l_self_modules_blocks_modules_3_modules_2_modules_conv_dw_modules_3_parameters_weight_ = (None)
        x_87 = torch.cat([conv2d_66, conv2d_67, conv2d_68, conv2d_69], 1)
        conv2d_66 = conv2d_67 = conv2d_68 = conv2d_69 = None
        x_88 = torch.nn.functional.batch_norm(
            x_87,
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_87 = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_bias_
        ) = None
        x_89 = torch.nn.functional.silu(x_88, inplace=True)
        x_88 = None
        x_se_24 = x_89.mean((2, 3), keepdim=True)
        x_se_25 = torch.conv2d(
            x_se_24,
            l_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_24 = l_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_26 = torch.nn.functional.silu(x_se_25, inplace=True)
        x_se_25 = None
        x_se_27 = torch.conv2d(
            x_se_26,
            l_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_26 = l_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_6 = torch.sigmoid(x_se_27)
        x_se_27 = None
        x_90 = x_89 * sigmoid_6
        x_89 = sigmoid_6 = None
        split_21 = torch.functional.split(x_90, [312, 312], 1)
        x_90 = None
        getitem_50 = split_21[0]
        getitem_51 = split_21[1]
        split_21 = None
        conv2d_72 = torch.conv2d(
            getitem_50,
            l_self_modules_blocks_modules_3_modules_2_modules_conv_pwl_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        getitem_50 = l_self_modules_blocks_modules_3_modules_2_modules_conv_pwl_modules_0_parameters_weight_ = (None)
        conv2d_73 = torch.conv2d(
            getitem_51,
            l_self_modules_blocks_modules_3_modules_2_modules_conv_pwl_modules_1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        getitem_51 = l_self_modules_blocks_modules_3_modules_2_modules_conv_pwl_modules_1_parameters_weight_ = (None)
        x_91 = torch.cat([conv2d_72, conv2d_73], 1)
        conv2d_72 = conv2d_73 = None
        x_92 = torch.nn.functional.batch_norm(
            x_91,
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_91 = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_bias_
        ) = None
        x_93 = x_92 + x_83
        x_92 = x_83 = None
        split_22 = torch.functional.split(x_93, [52, 52], 1)
        getitem_52 = split_22[0]
        getitem_53 = split_22[1]
        split_22 = None
        conv2d_74 = torch.conv2d(
            getitem_52,
            l_self_modules_blocks_modules_3_modules_3_modules_conv_pw_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        getitem_52 = l_self_modules_blocks_modules_3_modules_3_modules_conv_pw_modules_0_parameters_weight_ = (None)
        conv2d_75 = torch.conv2d(
            getitem_53,
            l_self_modules_blocks_modules_3_modules_3_modules_conv_pw_modules_1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        getitem_53 = l_self_modules_blocks_modules_3_modules_3_modules_conv_pw_modules_1_parameters_weight_ = (None)
        x_94 = torch.cat([conv2d_74, conv2d_75], 1)
        conv2d_74 = conv2d_75 = None
        x_95 = torch.nn.functional.batch_norm(
            x_94,
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_94 = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_bias_
        ) = None
        x_96 = torch.nn.functional.silu(x_95, inplace=True)
        x_95 = None
        split_23 = torch.functional.split(x_96, [156, 156, 156, 156], 1)
        x_96 = None
        getitem_54 = split_23[0]
        getitem_55 = split_23[1]
        getitem_56 = split_23[2]
        getitem_57 = split_23[3]
        split_23 = None
        conv2d_76 = torch.conv2d(
            getitem_54,
            l_self_modules_blocks_modules_3_modules_3_modules_conv_dw_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            156,
        )
        getitem_54 = l_self_modules_blocks_modules_3_modules_3_modules_conv_dw_modules_0_parameters_weight_ = (None)
        conv2d_77 = torch.conv2d(
            getitem_55,
            l_self_modules_blocks_modules_3_modules_3_modules_conv_dw_modules_1_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            156,
        )
        getitem_55 = l_self_modules_blocks_modules_3_modules_3_modules_conv_dw_modules_1_parameters_weight_ = (None)
        conv2d_78 = torch.conv2d(
            getitem_56,
            l_self_modules_blocks_modules_3_modules_3_modules_conv_dw_modules_2_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            156,
        )
        getitem_56 = l_self_modules_blocks_modules_3_modules_3_modules_conv_dw_modules_2_parameters_weight_ = (None)
        conv2d_79 = torch.conv2d(
            getitem_57,
            l_self_modules_blocks_modules_3_modules_3_modules_conv_dw_modules_3_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (1, 1),
            156,
        )
        getitem_57 = l_self_modules_blocks_modules_3_modules_3_modules_conv_dw_modules_3_parameters_weight_ = (None)
        x_97 = torch.cat([conv2d_76, conv2d_77, conv2d_78, conv2d_79], 1)
        conv2d_76 = conv2d_77 = conv2d_78 = conv2d_79 = None
        x_98 = torch.nn.functional.batch_norm(
            x_97,
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_97 = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_bias_
        ) = None
        x_99 = torch.nn.functional.silu(x_98, inplace=True)
        x_98 = None
        x_se_28 = x_99.mean((2, 3), keepdim=True)
        x_se_29 = torch.conv2d(
            x_se_28,
            l_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_28 = l_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_30 = torch.nn.functional.silu(x_se_29, inplace=True)
        x_se_29 = None
        x_se_31 = torch.conv2d(
            x_se_30,
            l_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_30 = l_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_7 = torch.sigmoid(x_se_31)
        x_se_31 = None
        x_100 = x_99 * sigmoid_7
        x_99 = sigmoid_7 = None
        split_24 = torch.functional.split(x_100, [312, 312], 1)
        x_100 = None
        getitem_58 = split_24[0]
        getitem_59 = split_24[1]
        split_24 = None
        conv2d_82 = torch.conv2d(
            getitem_58,
            l_self_modules_blocks_modules_3_modules_3_modules_conv_pwl_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        getitem_58 = l_self_modules_blocks_modules_3_modules_3_modules_conv_pwl_modules_0_parameters_weight_ = (None)
        conv2d_83 = torch.conv2d(
            getitem_59,
            l_self_modules_blocks_modules_3_modules_3_modules_conv_pwl_modules_1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        getitem_59 = l_self_modules_blocks_modules_3_modules_3_modules_conv_pwl_modules_1_parameters_weight_ = (None)
        x_101 = torch.cat([conv2d_82, conv2d_83], 1)
        conv2d_82 = conv2d_83 = None
        x_102 = torch.nn.functional.batch_norm(
            x_101,
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_101 = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_bias_
        ) = None
        x_103 = x_102 + x_93
        x_102 = x_93 = None
        x_104 = torch.conv2d(
            x_103,
            l_self_modules_blocks_modules_4_modules_0_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_103 = (
            l_self_modules_blocks_modules_4_modules_0_modules_conv_pw_parameters_weight_
        ) = None
        x_105 = torch.nn.functional.batch_norm(
            x_104,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_104 = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_106 = torch.nn.functional.silu(x_105, inplace=True)
        x_105 = None
        x_107 = torch.conv2d(
            x_106,
            l_self_modules_blocks_modules_4_modules_0_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            624,
        )
        x_106 = (
            l_self_modules_blocks_modules_4_modules_0_modules_conv_dw_parameters_weight_
        ) = None
        x_108 = torch.nn.functional.batch_norm(
            x_107,
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_107 = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_bias_
        ) = None
        x_109 = torch.nn.functional.silu(x_108, inplace=True)
        x_108 = None
        x_se_32 = x_109.mean((2, 3), keepdim=True)
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
        x_110 = x_109 * sigmoid_8
        x_109 = sigmoid_8 = None
        x_111 = torch.conv2d(
            x_110,
            l_self_modules_blocks_modules_4_modules_0_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_110 = l_self_modules_blocks_modules_4_modules_0_modules_conv_pwl_parameters_weight_ = (None)
        x_112 = torch.nn.functional.batch_norm(
            x_111,
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_111 = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_bias_
        ) = None
        split_25 = torch.functional.split(x_112, [80, 80], 1)
        getitem_60 = split_25[0]
        getitem_61 = split_25[1]
        split_25 = None
        conv2d_89 = torch.conv2d(
            getitem_60,
            l_self_modules_blocks_modules_4_modules_1_modules_conv_pw_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        getitem_60 = l_self_modules_blocks_modules_4_modules_1_modules_conv_pw_modules_0_parameters_weight_ = (None)
        conv2d_90 = torch.conv2d(
            getitem_61,
            l_self_modules_blocks_modules_4_modules_1_modules_conv_pw_modules_1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        getitem_61 = l_self_modules_blocks_modules_4_modules_1_modules_conv_pw_modules_1_parameters_weight_ = (None)
        x_113 = torch.cat([conv2d_89, conv2d_90], 1)
        conv2d_89 = conv2d_90 = None
        x_114 = torch.nn.functional.batch_norm(
            x_113,
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_113 = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_bias_
        ) = None
        x_115 = torch.nn.functional.silu(x_114, inplace=True)
        x_114 = None
        split_26 = torch.functional.split(x_115, [120, 120, 120, 120], 1)
        x_115 = None
        getitem_62 = split_26[0]
        getitem_63 = split_26[1]
        getitem_64 = split_26[2]
        getitem_65 = split_26[3]
        split_26 = None
        conv2d_91 = torch.conv2d(
            getitem_62,
            l_self_modules_blocks_modules_4_modules_1_modules_conv_dw_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            120,
        )
        getitem_62 = l_self_modules_blocks_modules_4_modules_1_modules_conv_dw_modules_0_parameters_weight_ = (None)
        conv2d_92 = torch.conv2d(
            getitem_63,
            l_self_modules_blocks_modules_4_modules_1_modules_conv_dw_modules_1_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            120,
        )
        getitem_63 = l_self_modules_blocks_modules_4_modules_1_modules_conv_dw_modules_1_parameters_weight_ = (None)
        conv2d_93 = torch.conv2d(
            getitem_64,
            l_self_modules_blocks_modules_4_modules_1_modules_conv_dw_modules_2_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            120,
        )
        getitem_64 = l_self_modules_blocks_modules_4_modules_1_modules_conv_dw_modules_2_parameters_weight_ = (None)
        conv2d_94 = torch.conv2d(
            getitem_65,
            l_self_modules_blocks_modules_4_modules_1_modules_conv_dw_modules_3_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (1, 1),
            120,
        )
        getitem_65 = l_self_modules_blocks_modules_4_modules_1_modules_conv_dw_modules_3_parameters_weight_ = (None)
        x_116 = torch.cat([conv2d_91, conv2d_92, conv2d_93, conv2d_94], 1)
        conv2d_91 = conv2d_92 = conv2d_93 = conv2d_94 = None
        x_117 = torch.nn.functional.batch_norm(
            x_116,
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_116 = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_bias_
        ) = None
        x_118 = torch.nn.functional.silu(x_117, inplace=True)
        x_117 = None
        x_se_36 = x_118.mean((2, 3), keepdim=True)
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
        x_119 = x_118 * sigmoid_9
        x_118 = sigmoid_9 = None
        split_27 = torch.functional.split(x_119, [240, 240], 1)
        x_119 = None
        getitem_66 = split_27[0]
        getitem_67 = split_27[1]
        split_27 = None
        conv2d_97 = torch.conv2d(
            getitem_66,
            l_self_modules_blocks_modules_4_modules_1_modules_conv_pwl_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        getitem_66 = l_self_modules_blocks_modules_4_modules_1_modules_conv_pwl_modules_0_parameters_weight_ = (None)
        conv2d_98 = torch.conv2d(
            getitem_67,
            l_self_modules_blocks_modules_4_modules_1_modules_conv_pwl_modules_1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        getitem_67 = l_self_modules_blocks_modules_4_modules_1_modules_conv_pwl_modules_1_parameters_weight_ = (None)
        x_120 = torch.cat([conv2d_97, conv2d_98], 1)
        conv2d_97 = conv2d_98 = None
        x_121 = torch.nn.functional.batch_norm(
            x_120,
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_120 = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_bias_
        ) = None
        x_122 = x_121 + x_112
        x_121 = x_112 = None
        split_28 = torch.functional.split(x_122, [80, 80], 1)
        getitem_68 = split_28[0]
        getitem_69 = split_28[1]
        split_28 = None
        conv2d_99 = torch.conv2d(
            getitem_68,
            l_self_modules_blocks_modules_4_modules_2_modules_conv_pw_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        getitem_68 = l_self_modules_blocks_modules_4_modules_2_modules_conv_pw_modules_0_parameters_weight_ = (None)
        conv2d_100 = torch.conv2d(
            getitem_69,
            l_self_modules_blocks_modules_4_modules_2_modules_conv_pw_modules_1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        getitem_69 = l_self_modules_blocks_modules_4_modules_2_modules_conv_pw_modules_1_parameters_weight_ = (None)
        x_123 = torch.cat([conv2d_99, conv2d_100], 1)
        conv2d_99 = conv2d_100 = None
        x_124 = torch.nn.functional.batch_norm(
            x_123,
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_123 = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_bias_
        ) = None
        x_125 = torch.nn.functional.silu(x_124, inplace=True)
        x_124 = None
        split_29 = torch.functional.split(x_125, [120, 120, 120, 120], 1)
        x_125 = None
        getitem_70 = split_29[0]
        getitem_71 = split_29[1]
        getitem_72 = split_29[2]
        getitem_73 = split_29[3]
        split_29 = None
        conv2d_101 = torch.conv2d(
            getitem_70,
            l_self_modules_blocks_modules_4_modules_2_modules_conv_dw_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            120,
        )
        getitem_70 = l_self_modules_blocks_modules_4_modules_2_modules_conv_dw_modules_0_parameters_weight_ = (None)
        conv2d_102 = torch.conv2d(
            getitem_71,
            l_self_modules_blocks_modules_4_modules_2_modules_conv_dw_modules_1_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            120,
        )
        getitem_71 = l_self_modules_blocks_modules_4_modules_2_modules_conv_dw_modules_1_parameters_weight_ = (None)
        conv2d_103 = torch.conv2d(
            getitem_72,
            l_self_modules_blocks_modules_4_modules_2_modules_conv_dw_modules_2_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            120,
        )
        getitem_72 = l_self_modules_blocks_modules_4_modules_2_modules_conv_dw_modules_2_parameters_weight_ = (None)
        conv2d_104 = torch.conv2d(
            getitem_73,
            l_self_modules_blocks_modules_4_modules_2_modules_conv_dw_modules_3_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (1, 1),
            120,
        )
        getitem_73 = l_self_modules_blocks_modules_4_modules_2_modules_conv_dw_modules_3_parameters_weight_ = (None)
        x_126 = torch.cat([conv2d_101, conv2d_102, conv2d_103, conv2d_104], 1)
        conv2d_101 = conv2d_102 = conv2d_103 = conv2d_104 = None
        x_127 = torch.nn.functional.batch_norm(
            x_126,
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_126 = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_bias_
        ) = None
        x_128 = torch.nn.functional.silu(x_127, inplace=True)
        x_127 = None
        x_se_40 = x_128.mean((2, 3), keepdim=True)
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
        x_129 = x_128 * sigmoid_10
        x_128 = sigmoid_10 = None
        split_30 = torch.functional.split(x_129, [240, 240], 1)
        x_129 = None
        getitem_74 = split_30[0]
        getitem_75 = split_30[1]
        split_30 = None
        conv2d_107 = torch.conv2d(
            getitem_74,
            l_self_modules_blocks_modules_4_modules_2_modules_conv_pwl_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        getitem_74 = l_self_modules_blocks_modules_4_modules_2_modules_conv_pwl_modules_0_parameters_weight_ = (None)
        conv2d_108 = torch.conv2d(
            getitem_75,
            l_self_modules_blocks_modules_4_modules_2_modules_conv_pwl_modules_1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        getitem_75 = l_self_modules_blocks_modules_4_modules_2_modules_conv_pwl_modules_1_parameters_weight_ = (None)
        x_130 = torch.cat([conv2d_107, conv2d_108], 1)
        conv2d_107 = conv2d_108 = None
        x_131 = torch.nn.functional.batch_norm(
            x_130,
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_130 = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_bias_
        ) = None
        x_132 = x_131 + x_122
        x_131 = x_122 = None
        split_31 = torch.functional.split(x_132, [80, 80], 1)
        getitem_76 = split_31[0]
        getitem_77 = split_31[1]
        split_31 = None
        conv2d_109 = torch.conv2d(
            getitem_76,
            l_self_modules_blocks_modules_4_modules_3_modules_conv_pw_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        getitem_76 = l_self_modules_blocks_modules_4_modules_3_modules_conv_pw_modules_0_parameters_weight_ = (None)
        conv2d_110 = torch.conv2d(
            getitem_77,
            l_self_modules_blocks_modules_4_modules_3_modules_conv_pw_modules_1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        getitem_77 = l_self_modules_blocks_modules_4_modules_3_modules_conv_pw_modules_1_parameters_weight_ = (None)
        x_133 = torch.cat([conv2d_109, conv2d_110], 1)
        conv2d_109 = conv2d_110 = None
        x_134 = torch.nn.functional.batch_norm(
            x_133,
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_133 = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_bias_
        ) = None
        x_135 = torch.nn.functional.silu(x_134, inplace=True)
        x_134 = None
        split_32 = torch.functional.split(x_135, [120, 120, 120, 120], 1)
        x_135 = None
        getitem_78 = split_32[0]
        getitem_79 = split_32[1]
        getitem_80 = split_32[2]
        getitem_81 = split_32[3]
        split_32 = None
        conv2d_111 = torch.conv2d(
            getitem_78,
            l_self_modules_blocks_modules_4_modules_3_modules_conv_dw_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            120,
        )
        getitem_78 = l_self_modules_blocks_modules_4_modules_3_modules_conv_dw_modules_0_parameters_weight_ = (None)
        conv2d_112 = torch.conv2d(
            getitem_79,
            l_self_modules_blocks_modules_4_modules_3_modules_conv_dw_modules_1_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            120,
        )
        getitem_79 = l_self_modules_blocks_modules_4_modules_3_modules_conv_dw_modules_1_parameters_weight_ = (None)
        conv2d_113 = torch.conv2d(
            getitem_80,
            l_self_modules_blocks_modules_4_modules_3_modules_conv_dw_modules_2_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            120,
        )
        getitem_80 = l_self_modules_blocks_modules_4_modules_3_modules_conv_dw_modules_2_parameters_weight_ = (None)
        conv2d_114 = torch.conv2d(
            getitem_81,
            l_self_modules_blocks_modules_4_modules_3_modules_conv_dw_modules_3_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (1, 1),
            120,
        )
        getitem_81 = l_self_modules_blocks_modules_4_modules_3_modules_conv_dw_modules_3_parameters_weight_ = (None)
        x_136 = torch.cat([conv2d_111, conv2d_112, conv2d_113, conv2d_114], 1)
        conv2d_111 = conv2d_112 = conv2d_113 = conv2d_114 = None
        x_137 = torch.nn.functional.batch_norm(
            x_136,
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_136 = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_bias_
        ) = None
        x_138 = torch.nn.functional.silu(x_137, inplace=True)
        x_137 = None
        x_se_44 = x_138.mean((2, 3), keepdim=True)
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
        x_139 = x_138 * sigmoid_11
        x_138 = sigmoid_11 = None
        split_33 = torch.functional.split(x_139, [240, 240], 1)
        x_139 = None
        getitem_82 = split_33[0]
        getitem_83 = split_33[1]
        split_33 = None
        conv2d_117 = torch.conv2d(
            getitem_82,
            l_self_modules_blocks_modules_4_modules_3_modules_conv_pwl_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        getitem_82 = l_self_modules_blocks_modules_4_modules_3_modules_conv_pwl_modules_0_parameters_weight_ = (None)
        conv2d_118 = torch.conv2d(
            getitem_83,
            l_self_modules_blocks_modules_4_modules_3_modules_conv_pwl_modules_1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        getitem_83 = l_self_modules_blocks_modules_4_modules_3_modules_conv_pwl_modules_1_parameters_weight_ = (None)
        x_140 = torch.cat([conv2d_117, conv2d_118], 1)
        conv2d_117 = conv2d_118 = None
        x_141 = torch.nn.functional.batch_norm(
            x_140,
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_140 = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_bias_
        ) = None
        x_142 = x_141 + x_132
        x_141 = x_132 = None
        x_143 = torch.conv2d(
            x_142,
            l_self_modules_blocks_modules_5_modules_0_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_142 = (
            l_self_modules_blocks_modules_5_modules_0_modules_conv_pw_parameters_weight_
        ) = None
        x_144 = torch.nn.functional.batch_norm(
            x_143,
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_143 = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_145 = torch.nn.functional.silu(x_144, inplace=True)
        x_144 = None
        split_34 = torch.functional.split(x_145, [240, 240, 240, 240], 1)
        x_145 = None
        getitem_84 = split_34[0]
        getitem_85 = split_34[1]
        getitem_86 = split_34[2]
        getitem_87 = split_34[3]
        split_34 = None
        conv2d_120 = torch.conv2d(
            getitem_84,
            l_self_modules_blocks_modules_5_modules_0_modules_conv_dw_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            240,
        )
        getitem_84 = l_self_modules_blocks_modules_5_modules_0_modules_conv_dw_modules_0_parameters_weight_ = (None)
        conv2d_121 = torch.conv2d(
            getitem_85,
            l_self_modules_blocks_modules_5_modules_0_modules_conv_dw_modules_1_parameters_weight_,
            None,
            (2, 2),
            (2, 2),
            (1, 1),
            240,
        )
        getitem_85 = l_self_modules_blocks_modules_5_modules_0_modules_conv_dw_modules_1_parameters_weight_ = (None)
        conv2d_122 = torch.conv2d(
            getitem_86,
            l_self_modules_blocks_modules_5_modules_0_modules_conv_dw_modules_2_parameters_weight_,
            None,
            (2, 2),
            (3, 3),
            (1, 1),
            240,
        )
        getitem_86 = l_self_modules_blocks_modules_5_modules_0_modules_conv_dw_modules_2_parameters_weight_ = (None)
        conv2d_123 = torch.conv2d(
            getitem_87,
            l_self_modules_blocks_modules_5_modules_0_modules_conv_dw_modules_3_parameters_weight_,
            None,
            (2, 2),
            (4, 4),
            (1, 1),
            240,
        )
        getitem_87 = l_self_modules_blocks_modules_5_modules_0_modules_conv_dw_modules_3_parameters_weight_ = (None)
        x_146 = torch.cat([conv2d_120, conv2d_121, conv2d_122, conv2d_123], 1)
        conv2d_120 = conv2d_121 = conv2d_122 = conv2d_123 = None
        x_147 = torch.nn.functional.batch_norm(
            x_146,
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_146 = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_bias_
        ) = None
        x_148 = torch.nn.functional.silu(x_147, inplace=True)
        x_147 = None
        x_se_48 = x_148.mean((2, 3), keepdim=True)
        x_se_49 = torch.conv2d(
            x_se_48,
            l_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_48 = l_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_50 = torch.nn.functional.silu(x_se_49, inplace=True)
        x_se_49 = None
        x_se_51 = torch.conv2d(
            x_se_50,
            l_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_50 = l_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_12 = torch.sigmoid(x_se_51)
        x_se_51 = None
        x_149 = x_148 * sigmoid_12
        x_148 = sigmoid_12 = None
        x_150 = torch.conv2d(
            x_149,
            l_self_modules_blocks_modules_5_modules_0_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_149 = l_self_modules_blocks_modules_5_modules_0_modules_conv_pwl_parameters_weight_ = (None)
        x_151 = torch.nn.functional.batch_norm(
            x_150,
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_150 = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_bias_
        ) = None
        x_152 = torch.conv2d(
            x_151,
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
        x_153 = torch.nn.functional.batch_norm(
            x_152,
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_152 = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_bias_
        ) = None
        x_154 = torch.nn.functional.silu(x_153, inplace=True)
        x_153 = None
        split_35 = torch.functional.split(x_154, [396, 396, 396, 396], 1)
        x_154 = None
        getitem_88 = split_35[0]
        getitem_89 = split_35[1]
        getitem_90 = split_35[2]
        getitem_91 = split_35[3]
        split_35 = None
        conv2d_128 = torch.conv2d(
            getitem_88,
            l_self_modules_blocks_modules_5_modules_1_modules_conv_dw_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            396,
        )
        getitem_88 = l_self_modules_blocks_modules_5_modules_1_modules_conv_dw_modules_0_parameters_weight_ = (None)
        conv2d_129 = torch.conv2d(
            getitem_89,
            l_self_modules_blocks_modules_5_modules_1_modules_conv_dw_modules_1_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            396,
        )
        getitem_89 = l_self_modules_blocks_modules_5_modules_1_modules_conv_dw_modules_1_parameters_weight_ = (None)
        conv2d_130 = torch.conv2d(
            getitem_90,
            l_self_modules_blocks_modules_5_modules_1_modules_conv_dw_modules_2_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            396,
        )
        getitem_90 = l_self_modules_blocks_modules_5_modules_1_modules_conv_dw_modules_2_parameters_weight_ = (None)
        conv2d_131 = torch.conv2d(
            getitem_91,
            l_self_modules_blocks_modules_5_modules_1_modules_conv_dw_modules_3_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (1, 1),
            396,
        )
        getitem_91 = l_self_modules_blocks_modules_5_modules_1_modules_conv_dw_modules_3_parameters_weight_ = (None)
        x_155 = torch.cat([conv2d_128, conv2d_129, conv2d_130, conv2d_131], 1)
        conv2d_128 = conv2d_129 = conv2d_130 = conv2d_131 = None
        x_156 = torch.nn.functional.batch_norm(
            x_155,
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_155 = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_bias_
        ) = None
        x_157 = torch.nn.functional.silu(x_156, inplace=True)
        x_156 = None
        x_se_52 = x_157.mean((2, 3), keepdim=True)
        x_se_53 = torch.conv2d(
            x_se_52,
            l_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_52 = l_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_54 = torch.nn.functional.silu(x_se_53, inplace=True)
        x_se_53 = None
        x_se_55 = torch.conv2d(
            x_se_54,
            l_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_54 = l_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_13 = torch.sigmoid(x_se_55)
        x_se_55 = None
        x_158 = x_157 * sigmoid_13
        x_157 = sigmoid_13 = None
        split_36 = torch.functional.split(x_158, [792, 792], 1)
        x_158 = None
        getitem_92 = split_36[0]
        getitem_93 = split_36[1]
        split_36 = None
        conv2d_134 = torch.conv2d(
            getitem_92,
            l_self_modules_blocks_modules_5_modules_1_modules_conv_pwl_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        getitem_92 = l_self_modules_blocks_modules_5_modules_1_modules_conv_pwl_modules_0_parameters_weight_ = (None)
        conv2d_135 = torch.conv2d(
            getitem_93,
            l_self_modules_blocks_modules_5_modules_1_modules_conv_pwl_modules_1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        getitem_93 = l_self_modules_blocks_modules_5_modules_1_modules_conv_pwl_modules_1_parameters_weight_ = (None)
        x_159 = torch.cat([conv2d_134, conv2d_135], 1)
        conv2d_134 = conv2d_135 = None
        x_160 = torch.nn.functional.batch_norm(
            x_159,
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_159 = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_bias_
        ) = None
        x_161 = x_160 + x_151
        x_160 = x_151 = None
        x_162 = torch.conv2d(
            x_161,
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
        x_163 = torch.nn.functional.batch_norm(
            x_162,
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_162 = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_bias_
        ) = None
        x_164 = torch.nn.functional.silu(x_163, inplace=True)
        x_163 = None
        split_37 = torch.functional.split(x_164, [396, 396, 396, 396], 1)
        x_164 = None
        getitem_94 = split_37[0]
        getitem_95 = split_37[1]
        getitem_96 = split_37[2]
        getitem_97 = split_37[3]
        split_37 = None
        conv2d_137 = torch.conv2d(
            getitem_94,
            l_self_modules_blocks_modules_5_modules_2_modules_conv_dw_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            396,
        )
        getitem_94 = l_self_modules_blocks_modules_5_modules_2_modules_conv_dw_modules_0_parameters_weight_ = (None)
        conv2d_138 = torch.conv2d(
            getitem_95,
            l_self_modules_blocks_modules_5_modules_2_modules_conv_dw_modules_1_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            396,
        )
        getitem_95 = l_self_modules_blocks_modules_5_modules_2_modules_conv_dw_modules_1_parameters_weight_ = (None)
        conv2d_139 = torch.conv2d(
            getitem_96,
            l_self_modules_blocks_modules_5_modules_2_modules_conv_dw_modules_2_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            396,
        )
        getitem_96 = l_self_modules_blocks_modules_5_modules_2_modules_conv_dw_modules_2_parameters_weight_ = (None)
        conv2d_140 = torch.conv2d(
            getitem_97,
            l_self_modules_blocks_modules_5_modules_2_modules_conv_dw_modules_3_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (1, 1),
            396,
        )
        getitem_97 = l_self_modules_blocks_modules_5_modules_2_modules_conv_dw_modules_3_parameters_weight_ = (None)
        x_165 = torch.cat([conv2d_137, conv2d_138, conv2d_139, conv2d_140], 1)
        conv2d_137 = conv2d_138 = conv2d_139 = conv2d_140 = None
        x_166 = torch.nn.functional.batch_norm(
            x_165,
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_165 = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_bias_
        ) = None
        x_167 = torch.nn.functional.silu(x_166, inplace=True)
        x_166 = None
        x_se_56 = x_167.mean((2, 3), keepdim=True)
        x_se_57 = torch.conv2d(
            x_se_56,
            l_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_56 = l_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_58 = torch.nn.functional.silu(x_se_57, inplace=True)
        x_se_57 = None
        x_se_59 = torch.conv2d(
            x_se_58,
            l_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_58 = l_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_14 = torch.sigmoid(x_se_59)
        x_se_59 = None
        x_168 = x_167 * sigmoid_14
        x_167 = sigmoid_14 = None
        split_38 = torch.functional.split(x_168, [792, 792], 1)
        x_168 = None
        getitem_98 = split_38[0]
        getitem_99 = split_38[1]
        split_38 = None
        conv2d_143 = torch.conv2d(
            getitem_98,
            l_self_modules_blocks_modules_5_modules_2_modules_conv_pwl_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        getitem_98 = l_self_modules_blocks_modules_5_modules_2_modules_conv_pwl_modules_0_parameters_weight_ = (None)
        conv2d_144 = torch.conv2d(
            getitem_99,
            l_self_modules_blocks_modules_5_modules_2_modules_conv_pwl_modules_1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        getitem_99 = l_self_modules_blocks_modules_5_modules_2_modules_conv_pwl_modules_1_parameters_weight_ = (None)
        x_169 = torch.cat([conv2d_143, conv2d_144], 1)
        conv2d_143 = conv2d_144 = None
        x_170 = torch.nn.functional.batch_norm(
            x_169,
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_169 = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_bias_
        ) = None
        x_171 = x_170 + x_161
        x_170 = x_161 = None
        x_172 = torch.conv2d(
            x_171,
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
        x_173 = torch.nn.functional.batch_norm(
            x_172,
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_172 = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_bias_
        ) = None
        x_174 = torch.nn.functional.silu(x_173, inplace=True)
        x_173 = None
        split_39 = torch.functional.split(x_174, [396, 396, 396, 396], 1)
        x_174 = None
        getitem_100 = split_39[0]
        getitem_101 = split_39[1]
        getitem_102 = split_39[2]
        getitem_103 = split_39[3]
        split_39 = None
        conv2d_146 = torch.conv2d(
            getitem_100,
            l_self_modules_blocks_modules_5_modules_3_modules_conv_dw_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            396,
        )
        getitem_100 = l_self_modules_blocks_modules_5_modules_3_modules_conv_dw_modules_0_parameters_weight_ = (None)
        conv2d_147 = torch.conv2d(
            getitem_101,
            l_self_modules_blocks_modules_5_modules_3_modules_conv_dw_modules_1_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            396,
        )
        getitem_101 = l_self_modules_blocks_modules_5_modules_3_modules_conv_dw_modules_1_parameters_weight_ = (None)
        conv2d_148 = torch.conv2d(
            getitem_102,
            l_self_modules_blocks_modules_5_modules_3_modules_conv_dw_modules_2_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            396,
        )
        getitem_102 = l_self_modules_blocks_modules_5_modules_3_modules_conv_dw_modules_2_parameters_weight_ = (None)
        conv2d_149 = torch.conv2d(
            getitem_103,
            l_self_modules_blocks_modules_5_modules_3_modules_conv_dw_modules_3_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (1, 1),
            396,
        )
        getitem_103 = l_self_modules_blocks_modules_5_modules_3_modules_conv_dw_modules_3_parameters_weight_ = (None)
        x_175 = torch.cat([conv2d_146, conv2d_147, conv2d_148, conv2d_149], 1)
        conv2d_146 = conv2d_147 = conv2d_148 = conv2d_149 = None
        x_176 = torch.nn.functional.batch_norm(
            x_175,
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_175 = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_bias_
        ) = None
        x_177 = torch.nn.functional.silu(x_176, inplace=True)
        x_176 = None
        x_se_60 = x_177.mean((2, 3), keepdim=True)
        x_se_61 = torch.conv2d(
            x_se_60,
            l_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_60 = l_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_62 = torch.nn.functional.silu(x_se_61, inplace=True)
        x_se_61 = None
        x_se_63 = torch.conv2d(
            x_se_62,
            l_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_62 = l_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_15 = torch.sigmoid(x_se_63)
        x_se_63 = None
        x_178 = x_177 * sigmoid_15
        x_177 = sigmoid_15 = None
        split_40 = torch.functional.split(x_178, [792, 792], 1)
        x_178 = None
        getitem_104 = split_40[0]
        getitem_105 = split_40[1]
        split_40 = None
        conv2d_152 = torch.conv2d(
            getitem_104,
            l_self_modules_blocks_modules_5_modules_3_modules_conv_pwl_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        getitem_104 = l_self_modules_blocks_modules_5_modules_3_modules_conv_pwl_modules_0_parameters_weight_ = (None)
        conv2d_153 = torch.conv2d(
            getitem_105,
            l_self_modules_blocks_modules_5_modules_3_modules_conv_pwl_modules_1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        getitem_105 = l_self_modules_blocks_modules_5_modules_3_modules_conv_pwl_modules_1_parameters_weight_ = (None)
        x_179 = torch.cat([conv2d_152, conv2d_153], 1)
        conv2d_152 = conv2d_153 = None
        x_180 = torch.nn.functional.batch_norm(
            x_179,
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_179 = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_bias_
        ) = None
        x_181 = x_180 + x_171
        x_180 = x_171 = None
        x_182 = torch.conv2d(
            x_181,
            l_self_modules_conv_head_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_181 = l_self_modules_conv_head_parameters_weight_ = None
        x_183 = torch.nn.functional.batch_norm(
            x_182,
            l_self_modules_bn2_buffers_running_mean_,
            l_self_modules_bn2_buffers_running_var_,
            l_self_modules_bn2_parameters_weight_,
            l_self_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_182 = (
            l_self_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_bn2_parameters_weight_
        ) = l_self_modules_bn2_parameters_bias_ = None
        x_184 = torch.nn.functional.relu(x_183, inplace=True)
        x_183 = None
        x_185 = torch.nn.functional.adaptive_avg_pool2d(x_184, 1)
        x_184 = None
        x_186 = x_185.flatten(1, -1)
        x_185 = None
        x_187 = torch._C._nn.linear(
            x_186,
            l_self_modules_classifier_parameters_weight_,
            l_self_modules_classifier_parameters_bias_,
        )
        x_186 = (
            l_self_modules_classifier_parameters_weight_
        ) = l_self_modules_classifier_parameters_bias_ = None
        return (x_187,)
