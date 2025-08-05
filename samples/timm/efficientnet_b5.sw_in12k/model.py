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
        L_self_modules_blocks_modules_0_modules_1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_1_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_1_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_1_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_1_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_2_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_2_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_2_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_2_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_1_modules_2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_2_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_2_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_2_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_2_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_2_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_2_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_2_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_2_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_2_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_3_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_3_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_3_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_3_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_3_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_3_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_3_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_3_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_3_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_3_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_3_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_3_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_3_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_3_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_3_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_3_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_3_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_4_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_4_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_4_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_4_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_4_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_4_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_4_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_4_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_4_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_4_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_4_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_4_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_4_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_4_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_4_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_4_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_4_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_4_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_4_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_2_modules_2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_2_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_2_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_3_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_3_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_4_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_4_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_4_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_4_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_4_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_4_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_4_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_4_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_4_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_4_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_4_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_4_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_4_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_4_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_4_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_4_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_4_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_4_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_4_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_6_modules_1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_1_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_1_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_1_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_1_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_1_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_2_modules_se_modules_conv_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_2_modules_se_modules_conv_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_2_modules_se_modules_conv_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_2_modules_se_modules_conv_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_2_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_2_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_2_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_2_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_2_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_blocks_modules_0_modules_1_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_1_modules_conv_dw_parameters_weight_
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
        l_self_modules_blocks_modules_0_modules_1_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_0_modules_1_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_0_modules_1_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_0_modules_1_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_0_modules_1_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_0_modules_1_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_0_modules_1_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_0_modules_1_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_0_modules_1_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_1_modules_conv_pw_parameters_weight_
        )
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
        l_self_modules_blocks_modules_0_modules_2_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_2_modules_conv_dw_parameters_weight_
        )
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
        l_self_modules_blocks_modules_0_modules_2_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_0_modules_2_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_0_modules_2_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_0_modules_2_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_0_modules_2_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_0_modules_2_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_0_modules_2_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_0_modules_2_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_0_modules_2_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_2_modules_conv_pw_parameters_weight_
        )
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
        l_self_modules_blocks_modules_1_modules_2_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_2_modules_conv_pw_parameters_weight_
        )
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
        l_self_modules_blocks_modules_1_modules_2_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_2_modules_conv_dw_parameters_weight_
        )
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
        l_self_modules_blocks_modules_1_modules_2_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_1_modules_2_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_1_modules_2_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_1_modules_2_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_1_modules_2_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_1_modules_2_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_1_modules_2_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_1_modules_2_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_1_modules_2_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_1_modules_2_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_1_modules_2_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_1_modules_2_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_1_modules_2_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_1_modules_2_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_1_modules_2_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_2_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_2_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_2_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_3_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_3_modules_conv_pw_parameters_weight_
        )
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
        l_self_modules_blocks_modules_1_modules_3_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_3_modules_conv_dw_parameters_weight_
        )
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
        l_self_modules_blocks_modules_1_modules_3_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_1_modules_3_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_1_modules_3_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_1_modules_3_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_1_modules_3_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_1_modules_3_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_1_modules_3_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_1_modules_3_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_1_modules_3_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_1_modules_3_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_1_modules_3_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_1_modules_3_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_1_modules_3_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_1_modules_3_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_1_modules_3_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_3_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_3_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_3_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_4_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_4_modules_conv_pw_parameters_weight_
        )
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
        l_self_modules_blocks_modules_1_modules_4_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_4_modules_conv_dw_parameters_weight_
        )
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
        l_self_modules_blocks_modules_1_modules_4_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_1_modules_4_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_1_modules_4_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_1_modules_4_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_1_modules_4_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_1_modules_4_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_1_modules_4_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_1_modules_4_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_1_modules_4_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_1_modules_4_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_1_modules_4_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_1_modules_4_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_1_modules_4_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_1_modules_4_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_1_modules_4_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_4_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_4_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_4_modules_bn3_parameters_bias_
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
        l_self_modules_blocks_modules_2_modules_2_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_2_modules_conv_pw_parameters_weight_
        )
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
        l_self_modules_blocks_modules_2_modules_2_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_2_modules_conv_dw_parameters_weight_
        )
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
        l_self_modules_blocks_modules_2_modules_2_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_2_modules_2_modules_conv_pwl_parameters_weight_
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
        l_self_modules_blocks_modules_2_modules_3_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_3_modules_conv_pw_parameters_weight_
        )
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
        l_self_modules_blocks_modules_2_modules_3_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_3_modules_conv_dw_parameters_weight_
        )
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
        l_self_modules_blocks_modules_2_modules_3_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_2_modules_3_modules_conv_pwl_parameters_weight_
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
        l_self_modules_blocks_modules_2_modules_4_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_4_modules_conv_pw_parameters_weight_
        )
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
        l_self_modules_blocks_modules_2_modules_4_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_4_modules_conv_dw_parameters_weight_
        )
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
        l_self_modules_blocks_modules_2_modules_4_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_2_modules_4_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_2_modules_4_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_2_modules_4_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_2_modules_4_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_2_modules_4_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_2_modules_4_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_2_modules_4_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_2_modules_4_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_2_modules_4_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_2_modules_4_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_2_modules_4_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_2_modules_4_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_2_modules_4_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_2_modules_4_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_4_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_4_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_4_modules_bn3_parameters_bias_
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
        l_self_modules_blocks_modules_6_modules_1_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_1_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_1_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_6_modules_1_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_6_modules_1_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_6_modules_1_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_6_modules_1_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_1_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_1_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_6_modules_1_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_6_modules_1_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_1_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_1_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_6_modules_1_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_6_modules_1_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_6_modules_1_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_6_modules_1_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_1_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_1_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_6_modules_1_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_6_modules_1_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_6_modules_1_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_6_modules_1_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_6_modules_1_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_6_modules_1_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_6_modules_1_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_6_modules_1_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_6_modules_1_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_6_modules_1_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_6_modules_1_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_6_modules_1_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_6_modules_1_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_6_modules_1_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_6_modules_1_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_6_modules_1_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_1_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_1_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_6_modules_1_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_6_modules_2_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_2_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_2_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_6_modules_2_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_6_modules_2_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_6_modules_2_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_6_modules_2_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_2_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_2_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_6_modules_2_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_6_modules_2_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_2_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_2_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_6_modules_2_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_6_modules_2_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_6_modules_2_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_6_modules_2_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_2_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_2_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_6_modules_2_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_6_modules_2_modules_se_modules_conv_reduce_parameters_weight_ = L_self_modules_blocks_modules_6_modules_2_modules_se_modules_conv_reduce_parameters_weight_
        l_self_modules_blocks_modules_6_modules_2_modules_se_modules_conv_reduce_parameters_bias_ = L_self_modules_blocks_modules_6_modules_2_modules_se_modules_conv_reduce_parameters_bias_
        l_self_modules_blocks_modules_6_modules_2_modules_se_modules_conv_expand_parameters_weight_ = L_self_modules_blocks_modules_6_modules_2_modules_se_modules_conv_expand_parameters_weight_
        l_self_modules_blocks_modules_6_modules_2_modules_se_modules_conv_expand_parameters_bias_ = L_self_modules_blocks_modules_6_modules_2_modules_se_modules_conv_expand_parameters_bias_
        l_self_modules_blocks_modules_6_modules_2_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_6_modules_2_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_6_modules_2_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_6_modules_2_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_6_modules_2_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_6_modules_2_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_6_modules_2_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_2_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_2_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_6_modules_2_modules_bn3_parameters_bias_
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
            48,
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
            l_self_modules_blocks_modules_0_modules_1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            24,
        )
        l_self_modules_blocks_modules_0_modules_1_modules_conv_dw_parameters_weight_ = (
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
        x_se_4 = x_11.mean((2, 3), keepdim=True)
        x_se_5 = torch.conv2d(
            x_se_4,
            l_self_modules_blocks_modules_0_modules_1_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_1_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_4 = l_self_modules_blocks_modules_0_modules_1_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_0_modules_1_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_6 = torch.nn.functional.silu(x_se_5, inplace=True)
        x_se_5 = None
        x_se_7 = torch.conv2d(
            x_se_6,
            l_self_modules_blocks_modules_0_modules_1_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_1_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_6 = l_self_modules_blocks_modules_0_modules_1_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_0_modules_1_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_1 = torch.sigmoid(x_se_7)
        x_se_7 = None
        x_12 = x_11 * sigmoid_1
        x_11 = sigmoid_1 = None
        x_13 = torch.conv2d(
            x_12,
            l_self_modules_blocks_modules_0_modules_1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_12 = (
            l_self_modules_blocks_modules_0_modules_1_modules_conv_pw_parameters_weight_
        ) = None
        x_14 = torch.nn.functional.batch_norm(
            x_13,
            l_self_modules_blocks_modules_0_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_0_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_0_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_13 = (
            l_self_modules_blocks_modules_0_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_0_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_0_modules_1_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_1_modules_bn2_parameters_bias_
        ) = None
        x_15 = x_14 + x_8
        x_14 = x_8 = None
        x_16 = torch.conv2d(
            x_15,
            l_self_modules_blocks_modules_0_modules_2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            24,
        )
        l_self_modules_blocks_modules_0_modules_2_modules_conv_dw_parameters_weight_ = (
            None
        )
        x_17 = torch.nn.functional.batch_norm(
            x_16,
            l_self_modules_blocks_modules_0_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_0_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_0_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_16 = (
            l_self_modules_blocks_modules_0_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_0_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_0_modules_2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_2_modules_bn1_parameters_bias_
        ) = None
        x_18 = torch.nn.functional.silu(x_17, inplace=True)
        x_17 = None
        x_se_8 = x_18.mean((2, 3), keepdim=True)
        x_se_9 = torch.conv2d(
            x_se_8,
            l_self_modules_blocks_modules_0_modules_2_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_2_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_8 = l_self_modules_blocks_modules_0_modules_2_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_0_modules_2_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_10 = torch.nn.functional.silu(x_se_9, inplace=True)
        x_se_9 = None
        x_se_11 = torch.conv2d(
            x_se_10,
            l_self_modules_blocks_modules_0_modules_2_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_2_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_10 = l_self_modules_blocks_modules_0_modules_2_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_0_modules_2_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_2 = torch.sigmoid(x_se_11)
        x_se_11 = None
        x_19 = x_18 * sigmoid_2
        x_18 = sigmoid_2 = None
        x_20 = torch.conv2d(
            x_19,
            l_self_modules_blocks_modules_0_modules_2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_19 = (
            l_self_modules_blocks_modules_0_modules_2_modules_conv_pw_parameters_weight_
        ) = None
        x_21 = torch.nn.functional.batch_norm(
            x_20,
            l_self_modules_blocks_modules_0_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_0_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_0_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_20 = (
            l_self_modules_blocks_modules_0_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_0_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_0_modules_2_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_2_modules_bn2_parameters_bias_
        ) = None
        x_22 = x_21 + x_15
        x_21 = x_15 = None
        x_23 = torch.conv2d(
            x_22,
            l_self_modules_blocks_modules_1_modules_0_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_22 = (
            l_self_modules_blocks_modules_1_modules_0_modules_conv_pw_parameters_weight_
        ) = None
        x_24 = torch.nn.functional.batch_norm(
            x_23,
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_23 = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_25 = torch.nn.functional.silu(x_24, inplace=True)
        x_24 = None
        x_26 = torch.conv2d(
            x_25,
            l_self_modules_blocks_modules_1_modules_0_modules_conv_dw_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            144,
        )
        x_25 = (
            l_self_modules_blocks_modules_1_modules_0_modules_conv_dw_parameters_weight_
        ) = None
        x_27 = torch.nn.functional.batch_norm(
            x_26,
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_26 = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_bias_
        ) = None
        x_28 = torch.nn.functional.silu(x_27, inplace=True)
        x_27 = None
        x_se_12 = x_28.mean((2, 3), keepdim=True)
        x_se_13 = torch.conv2d(
            x_se_12,
            l_self_modules_blocks_modules_1_modules_0_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_12 = l_self_modules_blocks_modules_1_modules_0_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_1_modules_0_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_14 = torch.nn.functional.silu(x_se_13, inplace=True)
        x_se_13 = None
        x_se_15 = torch.conv2d(
            x_se_14,
            l_self_modules_blocks_modules_1_modules_0_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_14 = l_self_modules_blocks_modules_1_modules_0_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_1_modules_0_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_3 = torch.sigmoid(x_se_15)
        x_se_15 = None
        x_29 = x_28 * sigmoid_3
        x_28 = sigmoid_3 = None
        x_30 = torch.conv2d(
            x_29,
            l_self_modules_blocks_modules_1_modules_0_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_29 = l_self_modules_blocks_modules_1_modules_0_modules_conv_pwl_parameters_weight_ = (None)
        x_31 = torch.nn.functional.batch_norm(
            x_30,
            l_self_modules_blocks_modules_1_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_30 = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn3_parameters_bias_
        ) = None
        x_32 = torch.conv2d(
            x_31,
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
        x_33 = torch.nn.functional.batch_norm(
            x_32,
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_32 = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_bias_
        ) = None
        x_34 = torch.nn.functional.silu(x_33, inplace=True)
        x_33 = None
        x_35 = torch.conv2d(
            x_34,
            l_self_modules_blocks_modules_1_modules_1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            240,
        )
        x_34 = (
            l_self_modules_blocks_modules_1_modules_1_modules_conv_dw_parameters_weight_
        ) = None
        x_36 = torch.nn.functional.batch_norm(
            x_35,
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_35 = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_bias_
        ) = None
        x_37 = torch.nn.functional.silu(x_36, inplace=True)
        x_36 = None
        x_se_16 = x_37.mean((2, 3), keepdim=True)
        x_se_17 = torch.conv2d(
            x_se_16,
            l_self_modules_blocks_modules_1_modules_1_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_1_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_16 = l_self_modules_blocks_modules_1_modules_1_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_1_modules_1_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_18 = torch.nn.functional.silu(x_se_17, inplace=True)
        x_se_17 = None
        x_se_19 = torch.conv2d(
            x_se_18,
            l_self_modules_blocks_modules_1_modules_1_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_1_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_18 = l_self_modules_blocks_modules_1_modules_1_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_1_modules_1_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_4 = torch.sigmoid(x_se_19)
        x_se_19 = None
        x_38 = x_37 * sigmoid_4
        x_37 = sigmoid_4 = None
        x_39 = torch.conv2d(
            x_38,
            l_self_modules_blocks_modules_1_modules_1_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_38 = l_self_modules_blocks_modules_1_modules_1_modules_conv_pwl_parameters_weight_ = (None)
        x_40 = torch.nn.functional.batch_norm(
            x_39,
            l_self_modules_blocks_modules_1_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_39 = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn3_parameters_bias_
        ) = None
        x_41 = x_40 + x_31
        x_40 = x_31 = None
        x_42 = torch.conv2d(
            x_41,
            l_self_modules_blocks_modules_1_modules_2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_1_modules_2_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_43 = torch.nn.functional.batch_norm(
            x_42,
            l_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_42 = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_bias_
        ) = None
        x_44 = torch.nn.functional.silu(x_43, inplace=True)
        x_43 = None
        x_45 = torch.conv2d(
            x_44,
            l_self_modules_blocks_modules_1_modules_2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            240,
        )
        x_44 = (
            l_self_modules_blocks_modules_1_modules_2_modules_conv_dw_parameters_weight_
        ) = None
        x_46 = torch.nn.functional.batch_norm(
            x_45,
            l_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_45 = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_bias_
        ) = None
        x_47 = torch.nn.functional.silu(x_46, inplace=True)
        x_46 = None
        x_se_20 = x_47.mean((2, 3), keepdim=True)
        x_se_21 = torch.conv2d(
            x_se_20,
            l_self_modules_blocks_modules_1_modules_2_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_2_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_20 = l_self_modules_blocks_modules_1_modules_2_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_1_modules_2_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_22 = torch.nn.functional.silu(x_se_21, inplace=True)
        x_se_21 = None
        x_se_23 = torch.conv2d(
            x_se_22,
            l_self_modules_blocks_modules_1_modules_2_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_2_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_22 = l_self_modules_blocks_modules_1_modules_2_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_1_modules_2_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_5 = torch.sigmoid(x_se_23)
        x_se_23 = None
        x_48 = x_47 * sigmoid_5
        x_47 = sigmoid_5 = None
        x_49 = torch.conv2d(
            x_48,
            l_self_modules_blocks_modules_1_modules_2_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_48 = l_self_modules_blocks_modules_1_modules_2_modules_conv_pwl_parameters_weight_ = (None)
        x_50 = torch.nn.functional.batch_norm(
            x_49,
            l_self_modules_blocks_modules_1_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_49 = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn3_parameters_bias_
        ) = None
        x_51 = x_50 + x_41
        x_50 = x_41 = None
        x_52 = torch.conv2d(
            x_51,
            l_self_modules_blocks_modules_1_modules_3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_1_modules_3_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_53 = torch.nn.functional.batch_norm(
            x_52,
            l_self_modules_blocks_modules_1_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_52 = (
            l_self_modules_blocks_modules_1_modules_3_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_3_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_3_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_3_modules_bn1_parameters_bias_
        ) = None
        x_54 = torch.nn.functional.silu(x_53, inplace=True)
        x_53 = None
        x_55 = torch.conv2d(
            x_54,
            l_self_modules_blocks_modules_1_modules_3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            240,
        )
        x_54 = (
            l_self_modules_blocks_modules_1_modules_3_modules_conv_dw_parameters_weight_
        ) = None
        x_56 = torch.nn.functional.batch_norm(
            x_55,
            l_self_modules_blocks_modules_1_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_55 = (
            l_self_modules_blocks_modules_1_modules_3_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_3_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_3_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_3_modules_bn2_parameters_bias_
        ) = None
        x_57 = torch.nn.functional.silu(x_56, inplace=True)
        x_56 = None
        x_se_24 = x_57.mean((2, 3), keepdim=True)
        x_se_25 = torch.conv2d(
            x_se_24,
            l_self_modules_blocks_modules_1_modules_3_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_3_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_24 = l_self_modules_blocks_modules_1_modules_3_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_1_modules_3_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_26 = torch.nn.functional.silu(x_se_25, inplace=True)
        x_se_25 = None
        x_se_27 = torch.conv2d(
            x_se_26,
            l_self_modules_blocks_modules_1_modules_3_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_3_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_26 = l_self_modules_blocks_modules_1_modules_3_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_1_modules_3_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_6 = torch.sigmoid(x_se_27)
        x_se_27 = None
        x_58 = x_57 * sigmoid_6
        x_57 = sigmoid_6 = None
        x_59 = torch.conv2d(
            x_58,
            l_self_modules_blocks_modules_1_modules_3_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_58 = l_self_modules_blocks_modules_1_modules_3_modules_conv_pwl_parameters_weight_ = (None)
        x_60 = torch.nn.functional.batch_norm(
            x_59,
            l_self_modules_blocks_modules_1_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_59 = (
            l_self_modules_blocks_modules_1_modules_3_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_3_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_3_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_3_modules_bn3_parameters_bias_
        ) = None
        x_61 = x_60 + x_51
        x_60 = x_51 = None
        x_62 = torch.conv2d(
            x_61,
            l_self_modules_blocks_modules_1_modules_4_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_1_modules_4_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_63 = torch.nn.functional.batch_norm(
            x_62,
            l_self_modules_blocks_modules_1_modules_4_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_4_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_4_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_4_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_62 = (
            l_self_modules_blocks_modules_1_modules_4_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_4_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_4_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_4_modules_bn1_parameters_bias_
        ) = None
        x_64 = torch.nn.functional.silu(x_63, inplace=True)
        x_63 = None
        x_65 = torch.conv2d(
            x_64,
            l_self_modules_blocks_modules_1_modules_4_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            240,
        )
        x_64 = (
            l_self_modules_blocks_modules_1_modules_4_modules_conv_dw_parameters_weight_
        ) = None
        x_66 = torch.nn.functional.batch_norm(
            x_65,
            l_self_modules_blocks_modules_1_modules_4_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_4_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_4_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_4_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_65 = (
            l_self_modules_blocks_modules_1_modules_4_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_4_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_4_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_4_modules_bn2_parameters_bias_
        ) = None
        x_67 = torch.nn.functional.silu(x_66, inplace=True)
        x_66 = None
        x_se_28 = x_67.mean((2, 3), keepdim=True)
        x_se_29 = torch.conv2d(
            x_se_28,
            l_self_modules_blocks_modules_1_modules_4_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_4_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_28 = l_self_modules_blocks_modules_1_modules_4_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_1_modules_4_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_30 = torch.nn.functional.silu(x_se_29, inplace=True)
        x_se_29 = None
        x_se_31 = torch.conv2d(
            x_se_30,
            l_self_modules_blocks_modules_1_modules_4_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_4_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_30 = l_self_modules_blocks_modules_1_modules_4_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_1_modules_4_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_7 = torch.sigmoid(x_se_31)
        x_se_31 = None
        x_68 = x_67 * sigmoid_7
        x_67 = sigmoid_7 = None
        x_69 = torch.conv2d(
            x_68,
            l_self_modules_blocks_modules_1_modules_4_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_68 = l_self_modules_blocks_modules_1_modules_4_modules_conv_pwl_parameters_weight_ = (None)
        x_70 = torch.nn.functional.batch_norm(
            x_69,
            l_self_modules_blocks_modules_1_modules_4_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_4_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_4_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_4_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_69 = (
            l_self_modules_blocks_modules_1_modules_4_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_4_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_4_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_4_modules_bn3_parameters_bias_
        ) = None
        x_71 = x_70 + x_61
        x_70 = x_61 = None
        x_72 = torch.conv2d(
            x_71,
            l_self_modules_blocks_modules_2_modules_0_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_71 = (
            l_self_modules_blocks_modules_2_modules_0_modules_conv_pw_parameters_weight_
        ) = None
        x_73 = torch.nn.functional.batch_norm(
            x_72,
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_72 = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_74 = torch.nn.functional.silu(x_73, inplace=True)
        x_73 = None
        x_75 = torch.conv2d(
            x_74,
            l_self_modules_blocks_modules_2_modules_0_modules_conv_dw_parameters_weight_,
            None,
            (2, 2),
            (2, 2),
            (1, 1),
            240,
        )
        x_74 = (
            l_self_modules_blocks_modules_2_modules_0_modules_conv_dw_parameters_weight_
        ) = None
        x_76 = torch.nn.functional.batch_norm(
            x_75,
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_75 = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_bias_
        ) = None
        x_77 = torch.nn.functional.silu(x_76, inplace=True)
        x_76 = None
        x_se_32 = x_77.mean((2, 3), keepdim=True)
        x_se_33 = torch.conv2d(
            x_se_32,
            l_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_32 = l_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_34 = torch.nn.functional.silu(x_se_33, inplace=True)
        x_se_33 = None
        x_se_35 = torch.conv2d(
            x_se_34,
            l_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_34 = l_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_8 = torch.sigmoid(x_se_35)
        x_se_35 = None
        x_78 = x_77 * sigmoid_8
        x_77 = sigmoid_8 = None
        x_79 = torch.conv2d(
            x_78,
            l_self_modules_blocks_modules_2_modules_0_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_78 = l_self_modules_blocks_modules_2_modules_0_modules_conv_pwl_parameters_weight_ = (None)
        x_80 = torch.nn.functional.batch_norm(
            x_79,
            l_self_modules_blocks_modules_2_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_79 = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn3_parameters_bias_
        ) = None
        x_81 = torch.conv2d(
            x_80,
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
        x_82 = torch.nn.functional.batch_norm(
            x_81,
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_81 = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_bias_
        ) = None
        x_83 = torch.nn.functional.silu(x_82, inplace=True)
        x_82 = None
        x_84 = torch.conv2d(
            x_83,
            l_self_modules_blocks_modules_2_modules_1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        x_83 = (
            l_self_modules_blocks_modules_2_modules_1_modules_conv_dw_parameters_weight_
        ) = None
        x_85 = torch.nn.functional.batch_norm(
            x_84,
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_84 = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_bias_
        ) = None
        x_86 = torch.nn.functional.silu(x_85, inplace=True)
        x_85 = None
        x_se_36 = x_86.mean((2, 3), keepdim=True)
        x_se_37 = torch.conv2d(
            x_se_36,
            l_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_36 = l_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_38 = torch.nn.functional.silu(x_se_37, inplace=True)
        x_se_37 = None
        x_se_39 = torch.conv2d(
            x_se_38,
            l_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_38 = l_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_9 = torch.sigmoid(x_se_39)
        x_se_39 = None
        x_87 = x_86 * sigmoid_9
        x_86 = sigmoid_9 = None
        x_88 = torch.conv2d(
            x_87,
            l_self_modules_blocks_modules_2_modules_1_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_87 = l_self_modules_blocks_modules_2_modules_1_modules_conv_pwl_parameters_weight_ = (None)
        x_89 = torch.nn.functional.batch_norm(
            x_88,
            l_self_modules_blocks_modules_2_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_88 = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn3_parameters_bias_
        ) = None
        x_90 = x_89 + x_80
        x_89 = x_80 = None
        x_91 = torch.conv2d(
            x_90,
            l_self_modules_blocks_modules_2_modules_2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_2_modules_2_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_92 = torch.nn.functional.batch_norm(
            x_91,
            l_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_91 = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_bias_
        ) = None
        x_93 = torch.nn.functional.silu(x_92, inplace=True)
        x_92 = None
        x_94 = torch.conv2d(
            x_93,
            l_self_modules_blocks_modules_2_modules_2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        x_93 = (
            l_self_modules_blocks_modules_2_modules_2_modules_conv_dw_parameters_weight_
        ) = None
        x_95 = torch.nn.functional.batch_norm(
            x_94,
            l_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_94 = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_bias_
        ) = None
        x_96 = torch.nn.functional.silu(x_95, inplace=True)
        x_95 = None
        x_se_40 = x_96.mean((2, 3), keepdim=True)
        x_se_41 = torch.conv2d(
            x_se_40,
            l_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_40 = l_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_42 = torch.nn.functional.silu(x_se_41, inplace=True)
        x_se_41 = None
        x_se_43 = torch.conv2d(
            x_se_42,
            l_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_42 = l_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_10 = torch.sigmoid(x_se_43)
        x_se_43 = None
        x_97 = x_96 * sigmoid_10
        x_96 = sigmoid_10 = None
        x_98 = torch.conv2d(
            x_97,
            l_self_modules_blocks_modules_2_modules_2_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_97 = l_self_modules_blocks_modules_2_modules_2_modules_conv_pwl_parameters_weight_ = (None)
        x_99 = torch.nn.functional.batch_norm(
            x_98,
            l_self_modules_blocks_modules_2_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_98 = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn3_parameters_bias_
        ) = None
        x_100 = x_99 + x_90
        x_99 = x_90 = None
        x_101 = torch.conv2d(
            x_100,
            l_self_modules_blocks_modules_2_modules_3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_2_modules_3_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_102 = torch.nn.functional.batch_norm(
            x_101,
            l_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_101 = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_bias_
        ) = None
        x_103 = torch.nn.functional.silu(x_102, inplace=True)
        x_102 = None
        x_104 = torch.conv2d(
            x_103,
            l_self_modules_blocks_modules_2_modules_3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        x_103 = (
            l_self_modules_blocks_modules_2_modules_3_modules_conv_dw_parameters_weight_
        ) = None
        x_105 = torch.nn.functional.batch_norm(
            x_104,
            l_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_104 = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_bias_
        ) = None
        x_106 = torch.nn.functional.silu(x_105, inplace=True)
        x_105 = None
        x_se_44 = x_106.mean((2, 3), keepdim=True)
        x_se_45 = torch.conv2d(
            x_se_44,
            l_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_44 = l_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_46 = torch.nn.functional.silu(x_se_45, inplace=True)
        x_se_45 = None
        x_se_47 = torch.conv2d(
            x_se_46,
            l_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_46 = l_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_11 = torch.sigmoid(x_se_47)
        x_se_47 = None
        x_107 = x_106 * sigmoid_11
        x_106 = sigmoid_11 = None
        x_108 = torch.conv2d(
            x_107,
            l_self_modules_blocks_modules_2_modules_3_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_107 = l_self_modules_blocks_modules_2_modules_3_modules_conv_pwl_parameters_weight_ = (None)
        x_109 = torch.nn.functional.batch_norm(
            x_108,
            l_self_modules_blocks_modules_2_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_108 = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn3_parameters_bias_
        ) = None
        x_110 = x_109 + x_100
        x_109 = x_100 = None
        x_111 = torch.conv2d(
            x_110,
            l_self_modules_blocks_modules_2_modules_4_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_2_modules_4_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_112 = torch.nn.functional.batch_norm(
            x_111,
            l_self_modules_blocks_modules_2_modules_4_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_4_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_4_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_4_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_111 = (
            l_self_modules_blocks_modules_2_modules_4_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_4_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_4_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_4_modules_bn1_parameters_bias_
        ) = None
        x_113 = torch.nn.functional.silu(x_112, inplace=True)
        x_112 = None
        x_114 = torch.conv2d(
            x_113,
            l_self_modules_blocks_modules_2_modules_4_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        x_113 = (
            l_self_modules_blocks_modules_2_modules_4_modules_conv_dw_parameters_weight_
        ) = None
        x_115 = torch.nn.functional.batch_norm(
            x_114,
            l_self_modules_blocks_modules_2_modules_4_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_4_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_4_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_4_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_114 = (
            l_self_modules_blocks_modules_2_modules_4_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_4_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_4_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_4_modules_bn2_parameters_bias_
        ) = None
        x_116 = torch.nn.functional.silu(x_115, inplace=True)
        x_115 = None
        x_se_48 = x_116.mean((2, 3), keepdim=True)
        x_se_49 = torch.conv2d(
            x_se_48,
            l_self_modules_blocks_modules_2_modules_4_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_4_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_48 = l_self_modules_blocks_modules_2_modules_4_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_2_modules_4_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_50 = torch.nn.functional.silu(x_se_49, inplace=True)
        x_se_49 = None
        x_se_51 = torch.conv2d(
            x_se_50,
            l_self_modules_blocks_modules_2_modules_4_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_4_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_50 = l_self_modules_blocks_modules_2_modules_4_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_2_modules_4_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_12 = torch.sigmoid(x_se_51)
        x_se_51 = None
        x_117 = x_116 * sigmoid_12
        x_116 = sigmoid_12 = None
        x_118 = torch.conv2d(
            x_117,
            l_self_modules_blocks_modules_2_modules_4_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_117 = l_self_modules_blocks_modules_2_modules_4_modules_conv_pwl_parameters_weight_ = (None)
        x_119 = torch.nn.functional.batch_norm(
            x_118,
            l_self_modules_blocks_modules_2_modules_4_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_4_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_4_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_4_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_118 = (
            l_self_modules_blocks_modules_2_modules_4_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_4_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_4_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_4_modules_bn3_parameters_bias_
        ) = None
        x_120 = x_119 + x_110
        x_119 = x_110 = None
        x_121 = torch.conv2d(
            x_120,
            l_self_modules_blocks_modules_3_modules_0_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_120 = (
            l_self_modules_blocks_modules_3_modules_0_modules_conv_pw_parameters_weight_
        ) = None
        x_122 = torch.nn.functional.batch_norm(
            x_121,
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_121 = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_123 = torch.nn.functional.silu(x_122, inplace=True)
        x_122 = None
        x_124 = torch.conv2d(
            x_123,
            l_self_modules_blocks_modules_3_modules_0_modules_conv_dw_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            384,
        )
        x_123 = (
            l_self_modules_blocks_modules_3_modules_0_modules_conv_dw_parameters_weight_
        ) = None
        x_125 = torch.nn.functional.batch_norm(
            x_124,
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_124 = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_bias_
        ) = None
        x_126 = torch.nn.functional.silu(x_125, inplace=True)
        x_125 = None
        x_se_52 = x_126.mean((2, 3), keepdim=True)
        x_se_53 = torch.conv2d(
            x_se_52,
            l_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_52 = l_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_54 = torch.nn.functional.silu(x_se_53, inplace=True)
        x_se_53 = None
        x_se_55 = torch.conv2d(
            x_se_54,
            l_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_54 = l_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_13 = torch.sigmoid(x_se_55)
        x_se_55 = None
        x_127 = x_126 * sigmoid_13
        x_126 = sigmoid_13 = None
        x_128 = torch.conv2d(
            x_127,
            l_self_modules_blocks_modules_3_modules_0_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_127 = l_self_modules_blocks_modules_3_modules_0_modules_conv_pwl_parameters_weight_ = (None)
        x_129 = torch.nn.functional.batch_norm(
            x_128,
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_128 = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_bias_
        ) = None
        x_130 = torch.conv2d(
            x_129,
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
        x_131 = torch.nn.functional.batch_norm(
            x_130,
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_130 = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_bias_
        ) = None
        x_132 = torch.nn.functional.silu(x_131, inplace=True)
        x_131 = None
        x_133 = torch.conv2d(
            x_132,
            l_self_modules_blocks_modules_3_modules_1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        x_132 = (
            l_self_modules_blocks_modules_3_modules_1_modules_conv_dw_parameters_weight_
        ) = None
        x_134 = torch.nn.functional.batch_norm(
            x_133,
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_133 = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_bias_
        ) = None
        x_135 = torch.nn.functional.silu(x_134, inplace=True)
        x_134 = None
        x_se_56 = x_135.mean((2, 3), keepdim=True)
        x_se_57 = torch.conv2d(
            x_se_56,
            l_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_56 = l_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_58 = torch.nn.functional.silu(x_se_57, inplace=True)
        x_se_57 = None
        x_se_59 = torch.conv2d(
            x_se_58,
            l_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_58 = l_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_14 = torch.sigmoid(x_se_59)
        x_se_59 = None
        x_136 = x_135 * sigmoid_14
        x_135 = sigmoid_14 = None
        x_137 = torch.conv2d(
            x_136,
            l_self_modules_blocks_modules_3_modules_1_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_136 = l_self_modules_blocks_modules_3_modules_1_modules_conv_pwl_parameters_weight_ = (None)
        x_138 = torch.nn.functional.batch_norm(
            x_137,
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_137 = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_bias_
        ) = None
        x_139 = x_138 + x_129
        x_138 = x_129 = None
        x_140 = torch.conv2d(
            x_139,
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
        x_141 = torch.nn.functional.batch_norm(
            x_140,
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_140 = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_bias_
        ) = None
        x_142 = torch.nn.functional.silu(x_141, inplace=True)
        x_141 = None
        x_143 = torch.conv2d(
            x_142,
            l_self_modules_blocks_modules_3_modules_2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        x_142 = (
            l_self_modules_blocks_modules_3_modules_2_modules_conv_dw_parameters_weight_
        ) = None
        x_144 = torch.nn.functional.batch_norm(
            x_143,
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_143 = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_bias_
        ) = None
        x_145 = torch.nn.functional.silu(x_144, inplace=True)
        x_144 = None
        x_se_60 = x_145.mean((2, 3), keepdim=True)
        x_se_61 = torch.conv2d(
            x_se_60,
            l_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_60 = l_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_62 = torch.nn.functional.silu(x_se_61, inplace=True)
        x_se_61 = None
        x_se_63 = torch.conv2d(
            x_se_62,
            l_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_62 = l_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_15 = torch.sigmoid(x_se_63)
        x_se_63 = None
        x_146 = x_145 * sigmoid_15
        x_145 = sigmoid_15 = None
        x_147 = torch.conv2d(
            x_146,
            l_self_modules_blocks_modules_3_modules_2_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_146 = l_self_modules_blocks_modules_3_modules_2_modules_conv_pwl_parameters_weight_ = (None)
        x_148 = torch.nn.functional.batch_norm(
            x_147,
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_147 = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_bias_
        ) = None
        x_149 = x_148 + x_139
        x_148 = x_139 = None
        x_150 = torch.conv2d(
            x_149,
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
        x_151 = torch.nn.functional.batch_norm(
            x_150,
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_150 = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_bias_
        ) = None
        x_152 = torch.nn.functional.silu(x_151, inplace=True)
        x_151 = None
        x_153 = torch.conv2d(
            x_152,
            l_self_modules_blocks_modules_3_modules_3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        x_152 = (
            l_self_modules_blocks_modules_3_modules_3_modules_conv_dw_parameters_weight_
        ) = None
        x_154 = torch.nn.functional.batch_norm(
            x_153,
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_153 = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_bias_
        ) = None
        x_155 = torch.nn.functional.silu(x_154, inplace=True)
        x_154 = None
        x_se_64 = x_155.mean((2, 3), keepdim=True)
        x_se_65 = torch.conv2d(
            x_se_64,
            l_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_64 = l_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_66 = torch.nn.functional.silu(x_se_65, inplace=True)
        x_se_65 = None
        x_se_67 = torch.conv2d(
            x_se_66,
            l_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_66 = l_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_16 = torch.sigmoid(x_se_67)
        x_se_67 = None
        x_156 = x_155 * sigmoid_16
        x_155 = sigmoid_16 = None
        x_157 = torch.conv2d(
            x_156,
            l_self_modules_blocks_modules_3_modules_3_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_156 = l_self_modules_blocks_modules_3_modules_3_modules_conv_pwl_parameters_weight_ = (None)
        x_158 = torch.nn.functional.batch_norm(
            x_157,
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_157 = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_bias_
        ) = None
        x_159 = x_158 + x_149
        x_158 = x_149 = None
        x_160 = torch.conv2d(
            x_159,
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
        x_161 = torch.nn.functional.batch_norm(
            x_160,
            l_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_160 = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_bias_
        ) = None
        x_162 = torch.nn.functional.silu(x_161, inplace=True)
        x_161 = None
        x_163 = torch.conv2d(
            x_162,
            l_self_modules_blocks_modules_3_modules_4_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        x_162 = (
            l_self_modules_blocks_modules_3_modules_4_modules_conv_dw_parameters_weight_
        ) = None
        x_164 = torch.nn.functional.batch_norm(
            x_163,
            l_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_163 = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_bias_
        ) = None
        x_165 = torch.nn.functional.silu(x_164, inplace=True)
        x_164 = None
        x_se_68 = x_165.mean((2, 3), keepdim=True)
        x_se_69 = torch.conv2d(
            x_se_68,
            l_self_modules_blocks_modules_3_modules_4_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_4_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_68 = l_self_modules_blocks_modules_3_modules_4_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_3_modules_4_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_70 = torch.nn.functional.silu(x_se_69, inplace=True)
        x_se_69 = None
        x_se_71 = torch.conv2d(
            x_se_70,
            l_self_modules_blocks_modules_3_modules_4_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_4_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_70 = l_self_modules_blocks_modules_3_modules_4_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_3_modules_4_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_17 = torch.sigmoid(x_se_71)
        x_se_71 = None
        x_166 = x_165 * sigmoid_17
        x_165 = sigmoid_17 = None
        x_167 = torch.conv2d(
            x_166,
            l_self_modules_blocks_modules_3_modules_4_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_166 = l_self_modules_blocks_modules_3_modules_4_modules_conv_pwl_parameters_weight_ = (None)
        x_168 = torch.nn.functional.batch_norm(
            x_167,
            l_self_modules_blocks_modules_3_modules_4_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_167 = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn3_parameters_bias_
        ) = None
        x_169 = x_168 + x_159
        x_168 = x_159 = None
        x_170 = torch.conv2d(
            x_169,
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
        x_171 = torch.nn.functional.batch_norm(
            x_170,
            l_self_modules_blocks_modules_3_modules_5_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_5_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_5_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_5_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_170 = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn1_parameters_bias_
        ) = None
        x_172 = torch.nn.functional.silu(x_171, inplace=True)
        x_171 = None
        x_173 = torch.conv2d(
            x_172,
            l_self_modules_blocks_modules_3_modules_5_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        x_172 = (
            l_self_modules_blocks_modules_3_modules_5_modules_conv_dw_parameters_weight_
        ) = None
        x_174 = torch.nn.functional.batch_norm(
            x_173,
            l_self_modules_blocks_modules_3_modules_5_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_5_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_5_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_5_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_173 = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn2_parameters_bias_
        ) = None
        x_175 = torch.nn.functional.silu(x_174, inplace=True)
        x_174 = None
        x_se_72 = x_175.mean((2, 3), keepdim=True)
        x_se_73 = torch.conv2d(
            x_se_72,
            l_self_modules_blocks_modules_3_modules_5_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_5_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_72 = l_self_modules_blocks_modules_3_modules_5_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_3_modules_5_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_74 = torch.nn.functional.silu(x_se_73, inplace=True)
        x_se_73 = None
        x_se_75 = torch.conv2d(
            x_se_74,
            l_self_modules_blocks_modules_3_modules_5_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_5_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_74 = l_self_modules_blocks_modules_3_modules_5_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_3_modules_5_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_18 = torch.sigmoid(x_se_75)
        x_se_75 = None
        x_176 = x_175 * sigmoid_18
        x_175 = sigmoid_18 = None
        x_177 = torch.conv2d(
            x_176,
            l_self_modules_blocks_modules_3_modules_5_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_176 = l_self_modules_blocks_modules_3_modules_5_modules_conv_pwl_parameters_weight_ = (None)
        x_178 = torch.nn.functional.batch_norm(
            x_177,
            l_self_modules_blocks_modules_3_modules_5_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_5_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_5_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_5_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_177 = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn3_parameters_bias_
        ) = None
        x_179 = x_178 + x_169
        x_178 = x_169 = None
        x_180 = torch.conv2d(
            x_179,
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
        x_181 = torch.nn.functional.batch_norm(
            x_180,
            l_self_modules_blocks_modules_3_modules_6_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_6_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_6_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_6_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_180 = (
            l_self_modules_blocks_modules_3_modules_6_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_6_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_6_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_6_modules_bn1_parameters_bias_
        ) = None
        x_182 = torch.nn.functional.silu(x_181, inplace=True)
        x_181 = None
        x_183 = torch.conv2d(
            x_182,
            l_self_modules_blocks_modules_3_modules_6_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        x_182 = (
            l_self_modules_blocks_modules_3_modules_6_modules_conv_dw_parameters_weight_
        ) = None
        x_184 = torch.nn.functional.batch_norm(
            x_183,
            l_self_modules_blocks_modules_3_modules_6_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_6_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_6_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_6_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_183 = (
            l_self_modules_blocks_modules_3_modules_6_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_6_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_6_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_6_modules_bn2_parameters_bias_
        ) = None
        x_185 = torch.nn.functional.silu(x_184, inplace=True)
        x_184 = None
        x_se_76 = x_185.mean((2, 3), keepdim=True)
        x_se_77 = torch.conv2d(
            x_se_76,
            l_self_modules_blocks_modules_3_modules_6_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_6_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_76 = l_self_modules_blocks_modules_3_modules_6_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_3_modules_6_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_78 = torch.nn.functional.silu(x_se_77, inplace=True)
        x_se_77 = None
        x_se_79 = torch.conv2d(
            x_se_78,
            l_self_modules_blocks_modules_3_modules_6_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_6_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_78 = l_self_modules_blocks_modules_3_modules_6_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_3_modules_6_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_19 = torch.sigmoid(x_se_79)
        x_se_79 = None
        x_186 = x_185 * sigmoid_19
        x_185 = sigmoid_19 = None
        x_187 = torch.conv2d(
            x_186,
            l_self_modules_blocks_modules_3_modules_6_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_186 = l_self_modules_blocks_modules_3_modules_6_modules_conv_pwl_parameters_weight_ = (None)
        x_188 = torch.nn.functional.batch_norm(
            x_187,
            l_self_modules_blocks_modules_3_modules_6_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_6_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_6_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_6_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_187 = (
            l_self_modules_blocks_modules_3_modules_6_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_6_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_6_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_6_modules_bn3_parameters_bias_
        ) = None
        x_189 = x_188 + x_179
        x_188 = x_179 = None
        x_190 = torch.conv2d(
            x_189,
            l_self_modules_blocks_modules_4_modules_0_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_189 = (
            l_self_modules_blocks_modules_4_modules_0_modules_conv_pw_parameters_weight_
        ) = None
        x_191 = torch.nn.functional.batch_norm(
            x_190,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_190 = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_192 = torch.nn.functional.silu(x_191, inplace=True)
        x_191 = None
        x_193 = torch.conv2d(
            x_192,
            l_self_modules_blocks_modules_4_modules_0_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            768,
        )
        x_192 = (
            l_self_modules_blocks_modules_4_modules_0_modules_conv_dw_parameters_weight_
        ) = None
        x_194 = torch.nn.functional.batch_norm(
            x_193,
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_193 = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_bias_
        ) = None
        x_195 = torch.nn.functional.silu(x_194, inplace=True)
        x_194 = None
        x_se_80 = x_195.mean((2, 3), keepdim=True)
        x_se_81 = torch.conv2d(
            x_se_80,
            l_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_80 = l_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_82 = torch.nn.functional.silu(x_se_81, inplace=True)
        x_se_81 = None
        x_se_83 = torch.conv2d(
            x_se_82,
            l_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_82 = l_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_20 = torch.sigmoid(x_se_83)
        x_se_83 = None
        x_196 = x_195 * sigmoid_20
        x_195 = sigmoid_20 = None
        x_197 = torch.conv2d(
            x_196,
            l_self_modules_blocks_modules_4_modules_0_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_196 = l_self_modules_blocks_modules_4_modules_0_modules_conv_pwl_parameters_weight_ = (None)
        x_198 = torch.nn.functional.batch_norm(
            x_197,
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_197 = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_bias_
        ) = None
        x_199 = torch.conv2d(
            x_198,
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
        x_200 = torch.nn.functional.batch_norm(
            x_199,
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_199 = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_bias_
        ) = None
        x_201 = torch.nn.functional.silu(x_200, inplace=True)
        x_200 = None
        x_202 = torch.conv2d(
            x_201,
            l_self_modules_blocks_modules_4_modules_1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1056,
        )
        x_201 = (
            l_self_modules_blocks_modules_4_modules_1_modules_conv_dw_parameters_weight_
        ) = None
        x_203 = torch.nn.functional.batch_norm(
            x_202,
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_202 = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_bias_
        ) = None
        x_204 = torch.nn.functional.silu(x_203, inplace=True)
        x_203 = None
        x_se_84 = x_204.mean((2, 3), keepdim=True)
        x_se_85 = torch.conv2d(
            x_se_84,
            l_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_84 = l_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_86 = torch.nn.functional.silu(x_se_85, inplace=True)
        x_se_85 = None
        x_se_87 = torch.conv2d(
            x_se_86,
            l_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_86 = l_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_21 = torch.sigmoid(x_se_87)
        x_se_87 = None
        x_205 = x_204 * sigmoid_21
        x_204 = sigmoid_21 = None
        x_206 = torch.conv2d(
            x_205,
            l_self_modules_blocks_modules_4_modules_1_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_205 = l_self_modules_blocks_modules_4_modules_1_modules_conv_pwl_parameters_weight_ = (None)
        x_207 = torch.nn.functional.batch_norm(
            x_206,
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_206 = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_bias_
        ) = None
        x_208 = x_207 + x_198
        x_207 = x_198 = None
        x_209 = torch.conv2d(
            x_208,
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
        x_210 = torch.nn.functional.batch_norm(
            x_209,
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_209 = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_bias_
        ) = None
        x_211 = torch.nn.functional.silu(x_210, inplace=True)
        x_210 = None
        x_212 = torch.conv2d(
            x_211,
            l_self_modules_blocks_modules_4_modules_2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1056,
        )
        x_211 = (
            l_self_modules_blocks_modules_4_modules_2_modules_conv_dw_parameters_weight_
        ) = None
        x_213 = torch.nn.functional.batch_norm(
            x_212,
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_212 = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_bias_
        ) = None
        x_214 = torch.nn.functional.silu(x_213, inplace=True)
        x_213 = None
        x_se_88 = x_214.mean((2, 3), keepdim=True)
        x_se_89 = torch.conv2d(
            x_se_88,
            l_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_88 = l_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_90 = torch.nn.functional.silu(x_se_89, inplace=True)
        x_se_89 = None
        x_se_91 = torch.conv2d(
            x_se_90,
            l_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_90 = l_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_22 = torch.sigmoid(x_se_91)
        x_se_91 = None
        x_215 = x_214 * sigmoid_22
        x_214 = sigmoid_22 = None
        x_216 = torch.conv2d(
            x_215,
            l_self_modules_blocks_modules_4_modules_2_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_215 = l_self_modules_blocks_modules_4_modules_2_modules_conv_pwl_parameters_weight_ = (None)
        x_217 = torch.nn.functional.batch_norm(
            x_216,
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_216 = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_bias_
        ) = None
        x_218 = x_217 + x_208
        x_217 = x_208 = None
        x_219 = torch.conv2d(
            x_218,
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
        x_220 = torch.nn.functional.batch_norm(
            x_219,
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_219 = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_bias_
        ) = None
        x_221 = torch.nn.functional.silu(x_220, inplace=True)
        x_220 = None
        x_222 = torch.conv2d(
            x_221,
            l_self_modules_blocks_modules_4_modules_3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1056,
        )
        x_221 = (
            l_self_modules_blocks_modules_4_modules_3_modules_conv_dw_parameters_weight_
        ) = None
        x_223 = torch.nn.functional.batch_norm(
            x_222,
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_222 = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_bias_
        ) = None
        x_224 = torch.nn.functional.silu(x_223, inplace=True)
        x_223 = None
        x_se_92 = x_224.mean((2, 3), keepdim=True)
        x_se_93 = torch.conv2d(
            x_se_92,
            l_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_92 = l_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_94 = torch.nn.functional.silu(x_se_93, inplace=True)
        x_se_93 = None
        x_se_95 = torch.conv2d(
            x_se_94,
            l_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_94 = l_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_23 = torch.sigmoid(x_se_95)
        x_se_95 = None
        x_225 = x_224 * sigmoid_23
        x_224 = sigmoid_23 = None
        x_226 = torch.conv2d(
            x_225,
            l_self_modules_blocks_modules_4_modules_3_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_225 = l_self_modules_blocks_modules_4_modules_3_modules_conv_pwl_parameters_weight_ = (None)
        x_227 = torch.nn.functional.batch_norm(
            x_226,
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_226 = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_bias_
        ) = None
        x_228 = x_227 + x_218
        x_227 = x_218 = None
        x_229 = torch.conv2d(
            x_228,
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
        x_230 = torch.nn.functional.batch_norm(
            x_229,
            l_self_modules_blocks_modules_4_modules_4_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_4_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_4_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_4_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_229 = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn1_parameters_bias_
        ) = None
        x_231 = torch.nn.functional.silu(x_230, inplace=True)
        x_230 = None
        x_232 = torch.conv2d(
            x_231,
            l_self_modules_blocks_modules_4_modules_4_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1056,
        )
        x_231 = (
            l_self_modules_blocks_modules_4_modules_4_modules_conv_dw_parameters_weight_
        ) = None
        x_233 = torch.nn.functional.batch_norm(
            x_232,
            l_self_modules_blocks_modules_4_modules_4_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_4_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_4_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_4_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_232 = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn2_parameters_bias_
        ) = None
        x_234 = torch.nn.functional.silu(x_233, inplace=True)
        x_233 = None
        x_se_96 = x_234.mean((2, 3), keepdim=True)
        x_se_97 = torch.conv2d(
            x_se_96,
            l_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_96 = l_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_98 = torch.nn.functional.silu(x_se_97, inplace=True)
        x_se_97 = None
        x_se_99 = torch.conv2d(
            x_se_98,
            l_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_98 = l_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_24 = torch.sigmoid(x_se_99)
        x_se_99 = None
        x_235 = x_234 * sigmoid_24
        x_234 = sigmoid_24 = None
        x_236 = torch.conv2d(
            x_235,
            l_self_modules_blocks_modules_4_modules_4_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_235 = l_self_modules_blocks_modules_4_modules_4_modules_conv_pwl_parameters_weight_ = (None)
        x_237 = torch.nn.functional.batch_norm(
            x_236,
            l_self_modules_blocks_modules_4_modules_4_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_4_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_4_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_4_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_236 = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn3_parameters_bias_
        ) = None
        x_238 = x_237 + x_228
        x_237 = x_228 = None
        x_239 = torch.conv2d(
            x_238,
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
        x_240 = torch.nn.functional.batch_norm(
            x_239,
            l_self_modules_blocks_modules_4_modules_5_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_5_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_5_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_5_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_239 = (
            l_self_modules_blocks_modules_4_modules_5_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_5_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_5_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_5_modules_bn1_parameters_bias_
        ) = None
        x_241 = torch.nn.functional.silu(x_240, inplace=True)
        x_240 = None
        x_242 = torch.conv2d(
            x_241,
            l_self_modules_blocks_modules_4_modules_5_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1056,
        )
        x_241 = (
            l_self_modules_blocks_modules_4_modules_5_modules_conv_dw_parameters_weight_
        ) = None
        x_243 = torch.nn.functional.batch_norm(
            x_242,
            l_self_modules_blocks_modules_4_modules_5_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_5_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_5_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_5_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_242 = (
            l_self_modules_blocks_modules_4_modules_5_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_5_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_5_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_5_modules_bn2_parameters_bias_
        ) = None
        x_244 = torch.nn.functional.silu(x_243, inplace=True)
        x_243 = None
        x_se_100 = x_244.mean((2, 3), keepdim=True)
        x_se_101 = torch.conv2d(
            x_se_100,
            l_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_100 = l_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_102 = torch.nn.functional.silu(x_se_101, inplace=True)
        x_se_101 = None
        x_se_103 = torch.conv2d(
            x_se_102,
            l_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_102 = l_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_25 = torch.sigmoid(x_se_103)
        x_se_103 = None
        x_245 = x_244 * sigmoid_25
        x_244 = sigmoid_25 = None
        x_246 = torch.conv2d(
            x_245,
            l_self_modules_blocks_modules_4_modules_5_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_245 = l_self_modules_blocks_modules_4_modules_5_modules_conv_pwl_parameters_weight_ = (None)
        x_247 = torch.nn.functional.batch_norm(
            x_246,
            l_self_modules_blocks_modules_4_modules_5_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_5_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_5_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_5_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_246 = (
            l_self_modules_blocks_modules_4_modules_5_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_5_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_5_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_5_modules_bn3_parameters_bias_
        ) = None
        x_248 = x_247 + x_238
        x_247 = x_238 = None
        x_249 = torch.conv2d(
            x_248,
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
        x_250 = torch.nn.functional.batch_norm(
            x_249,
            l_self_modules_blocks_modules_4_modules_6_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_6_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_6_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_6_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_249 = (
            l_self_modules_blocks_modules_4_modules_6_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_6_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_6_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_6_modules_bn1_parameters_bias_
        ) = None
        x_251 = torch.nn.functional.silu(x_250, inplace=True)
        x_250 = None
        x_252 = torch.conv2d(
            x_251,
            l_self_modules_blocks_modules_4_modules_6_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1056,
        )
        x_251 = (
            l_self_modules_blocks_modules_4_modules_6_modules_conv_dw_parameters_weight_
        ) = None
        x_253 = torch.nn.functional.batch_norm(
            x_252,
            l_self_modules_blocks_modules_4_modules_6_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_6_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_6_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_6_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_252 = (
            l_self_modules_blocks_modules_4_modules_6_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_6_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_6_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_6_modules_bn2_parameters_bias_
        ) = None
        x_254 = torch.nn.functional.silu(x_253, inplace=True)
        x_253 = None
        x_se_104 = x_254.mean((2, 3), keepdim=True)
        x_se_105 = torch.conv2d(
            x_se_104,
            l_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_104 = l_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_106 = torch.nn.functional.silu(x_se_105, inplace=True)
        x_se_105 = None
        x_se_107 = torch.conv2d(
            x_se_106,
            l_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_106 = l_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_26 = torch.sigmoid(x_se_107)
        x_se_107 = None
        x_255 = x_254 * sigmoid_26
        x_254 = sigmoid_26 = None
        x_256 = torch.conv2d(
            x_255,
            l_self_modules_blocks_modules_4_modules_6_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_255 = l_self_modules_blocks_modules_4_modules_6_modules_conv_pwl_parameters_weight_ = (None)
        x_257 = torch.nn.functional.batch_norm(
            x_256,
            l_self_modules_blocks_modules_4_modules_6_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_6_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_6_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_6_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_256 = (
            l_self_modules_blocks_modules_4_modules_6_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_6_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_6_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_6_modules_bn3_parameters_bias_
        ) = None
        x_258 = x_257 + x_248
        x_257 = x_248 = None
        x_259 = torch.conv2d(
            x_258,
            l_self_modules_blocks_modules_5_modules_0_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_258 = (
            l_self_modules_blocks_modules_5_modules_0_modules_conv_pw_parameters_weight_
        ) = None
        x_260 = torch.nn.functional.batch_norm(
            x_259,
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_259 = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_261 = torch.nn.functional.silu(x_260, inplace=True)
        x_260 = None
        x_262 = torch.conv2d(
            x_261,
            l_self_modules_blocks_modules_5_modules_0_modules_conv_dw_parameters_weight_,
            None,
            (2, 2),
            (2, 2),
            (1, 1),
            1056,
        )
        x_261 = (
            l_self_modules_blocks_modules_5_modules_0_modules_conv_dw_parameters_weight_
        ) = None
        x_263 = torch.nn.functional.batch_norm(
            x_262,
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_262 = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_bias_
        ) = None
        x_264 = torch.nn.functional.silu(x_263, inplace=True)
        x_263 = None
        x_se_108 = x_264.mean((2, 3), keepdim=True)
        x_se_109 = torch.conv2d(
            x_se_108,
            l_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_108 = l_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_110 = torch.nn.functional.silu(x_se_109, inplace=True)
        x_se_109 = None
        x_se_111 = torch.conv2d(
            x_se_110,
            l_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_110 = l_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_27 = torch.sigmoid(x_se_111)
        x_se_111 = None
        x_265 = x_264 * sigmoid_27
        x_264 = sigmoid_27 = None
        x_266 = torch.conv2d(
            x_265,
            l_self_modules_blocks_modules_5_modules_0_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_265 = l_self_modules_blocks_modules_5_modules_0_modules_conv_pwl_parameters_weight_ = (None)
        x_267 = torch.nn.functional.batch_norm(
            x_266,
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_266 = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_bias_
        ) = None
        x_268 = torch.conv2d(
            x_267,
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
        x_269 = torch.nn.functional.batch_norm(
            x_268,
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_268 = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_bias_
        ) = None
        x_270 = torch.nn.functional.silu(x_269, inplace=True)
        x_269 = None
        x_271 = torch.conv2d(
            x_270,
            l_self_modules_blocks_modules_5_modules_1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1824,
        )
        x_270 = (
            l_self_modules_blocks_modules_5_modules_1_modules_conv_dw_parameters_weight_
        ) = None
        x_272 = torch.nn.functional.batch_norm(
            x_271,
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_271 = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_bias_
        ) = None
        x_273 = torch.nn.functional.silu(x_272, inplace=True)
        x_272 = None
        x_se_112 = x_273.mean((2, 3), keepdim=True)
        x_se_113 = torch.conv2d(
            x_se_112,
            l_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_112 = l_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_114 = torch.nn.functional.silu(x_se_113, inplace=True)
        x_se_113 = None
        x_se_115 = torch.conv2d(
            x_se_114,
            l_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_114 = l_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_28 = torch.sigmoid(x_se_115)
        x_se_115 = None
        x_274 = x_273 * sigmoid_28
        x_273 = sigmoid_28 = None
        x_275 = torch.conv2d(
            x_274,
            l_self_modules_blocks_modules_5_modules_1_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_274 = l_self_modules_blocks_modules_5_modules_1_modules_conv_pwl_parameters_weight_ = (None)
        x_276 = torch.nn.functional.batch_norm(
            x_275,
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_275 = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_bias_
        ) = None
        x_277 = x_276 + x_267
        x_276 = x_267 = None
        x_278 = torch.conv2d(
            x_277,
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
        x_279 = torch.nn.functional.batch_norm(
            x_278,
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_278 = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_bias_
        ) = None
        x_280 = torch.nn.functional.silu(x_279, inplace=True)
        x_279 = None
        x_281 = torch.conv2d(
            x_280,
            l_self_modules_blocks_modules_5_modules_2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1824,
        )
        x_280 = (
            l_self_modules_blocks_modules_5_modules_2_modules_conv_dw_parameters_weight_
        ) = None
        x_282 = torch.nn.functional.batch_norm(
            x_281,
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_281 = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_bias_
        ) = None
        x_283 = torch.nn.functional.silu(x_282, inplace=True)
        x_282 = None
        x_se_116 = x_283.mean((2, 3), keepdim=True)
        x_se_117 = torch.conv2d(
            x_se_116,
            l_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_116 = l_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_118 = torch.nn.functional.silu(x_se_117, inplace=True)
        x_se_117 = None
        x_se_119 = torch.conv2d(
            x_se_118,
            l_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_118 = l_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_29 = torch.sigmoid(x_se_119)
        x_se_119 = None
        x_284 = x_283 * sigmoid_29
        x_283 = sigmoid_29 = None
        x_285 = torch.conv2d(
            x_284,
            l_self_modules_blocks_modules_5_modules_2_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_284 = l_self_modules_blocks_modules_5_modules_2_modules_conv_pwl_parameters_weight_ = (None)
        x_286 = torch.nn.functional.batch_norm(
            x_285,
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_285 = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_bias_
        ) = None
        x_287 = x_286 + x_277
        x_286 = x_277 = None
        x_288 = torch.conv2d(
            x_287,
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
        x_289 = torch.nn.functional.batch_norm(
            x_288,
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_288 = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_bias_
        ) = None
        x_290 = torch.nn.functional.silu(x_289, inplace=True)
        x_289 = None
        x_291 = torch.conv2d(
            x_290,
            l_self_modules_blocks_modules_5_modules_3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1824,
        )
        x_290 = (
            l_self_modules_blocks_modules_5_modules_3_modules_conv_dw_parameters_weight_
        ) = None
        x_292 = torch.nn.functional.batch_norm(
            x_291,
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_291 = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_bias_
        ) = None
        x_293 = torch.nn.functional.silu(x_292, inplace=True)
        x_292 = None
        x_se_120 = x_293.mean((2, 3), keepdim=True)
        x_se_121 = torch.conv2d(
            x_se_120,
            l_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_120 = l_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_122 = torch.nn.functional.silu(x_se_121, inplace=True)
        x_se_121 = None
        x_se_123 = torch.conv2d(
            x_se_122,
            l_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_122 = l_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_30 = torch.sigmoid(x_se_123)
        x_se_123 = None
        x_294 = x_293 * sigmoid_30
        x_293 = sigmoid_30 = None
        x_295 = torch.conv2d(
            x_294,
            l_self_modules_blocks_modules_5_modules_3_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_294 = l_self_modules_blocks_modules_5_modules_3_modules_conv_pwl_parameters_weight_ = (None)
        x_296 = torch.nn.functional.batch_norm(
            x_295,
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_295 = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_bias_
        ) = None
        x_297 = x_296 + x_287
        x_296 = x_287 = None
        x_298 = torch.conv2d(
            x_297,
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
        x_299 = torch.nn.functional.batch_norm(
            x_298,
            l_self_modules_blocks_modules_5_modules_4_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_4_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_4_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_4_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_298 = (
            l_self_modules_blocks_modules_5_modules_4_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_4_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_4_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_4_modules_bn1_parameters_bias_
        ) = None
        x_300 = torch.nn.functional.silu(x_299, inplace=True)
        x_299 = None
        x_301 = torch.conv2d(
            x_300,
            l_self_modules_blocks_modules_5_modules_4_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1824,
        )
        x_300 = (
            l_self_modules_blocks_modules_5_modules_4_modules_conv_dw_parameters_weight_
        ) = None
        x_302 = torch.nn.functional.batch_norm(
            x_301,
            l_self_modules_blocks_modules_5_modules_4_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_4_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_4_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_4_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_301 = (
            l_self_modules_blocks_modules_5_modules_4_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_4_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_4_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_4_modules_bn2_parameters_bias_
        ) = None
        x_303 = torch.nn.functional.silu(x_302, inplace=True)
        x_302 = None
        x_se_124 = x_303.mean((2, 3), keepdim=True)
        x_se_125 = torch.conv2d(
            x_se_124,
            l_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_124 = l_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_126 = torch.nn.functional.silu(x_se_125, inplace=True)
        x_se_125 = None
        x_se_127 = torch.conv2d(
            x_se_126,
            l_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_126 = l_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_31 = torch.sigmoid(x_se_127)
        x_se_127 = None
        x_304 = x_303 * sigmoid_31
        x_303 = sigmoid_31 = None
        x_305 = torch.conv2d(
            x_304,
            l_self_modules_blocks_modules_5_modules_4_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_304 = l_self_modules_blocks_modules_5_modules_4_modules_conv_pwl_parameters_weight_ = (None)
        x_306 = torch.nn.functional.batch_norm(
            x_305,
            l_self_modules_blocks_modules_5_modules_4_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_4_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_4_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_4_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_305 = (
            l_self_modules_blocks_modules_5_modules_4_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_4_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_4_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_4_modules_bn3_parameters_bias_
        ) = None
        x_307 = x_306 + x_297
        x_306 = x_297 = None
        x_308 = torch.conv2d(
            x_307,
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
        x_309 = torch.nn.functional.batch_norm(
            x_308,
            l_self_modules_blocks_modules_5_modules_5_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_5_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_5_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_5_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_308 = (
            l_self_modules_blocks_modules_5_modules_5_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_5_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_5_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_5_modules_bn1_parameters_bias_
        ) = None
        x_310 = torch.nn.functional.silu(x_309, inplace=True)
        x_309 = None
        x_311 = torch.conv2d(
            x_310,
            l_self_modules_blocks_modules_5_modules_5_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1824,
        )
        x_310 = (
            l_self_modules_blocks_modules_5_modules_5_modules_conv_dw_parameters_weight_
        ) = None
        x_312 = torch.nn.functional.batch_norm(
            x_311,
            l_self_modules_blocks_modules_5_modules_5_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_5_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_5_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_5_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_311 = (
            l_self_modules_blocks_modules_5_modules_5_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_5_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_5_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_5_modules_bn2_parameters_bias_
        ) = None
        x_313 = torch.nn.functional.silu(x_312, inplace=True)
        x_312 = None
        x_se_128 = x_313.mean((2, 3), keepdim=True)
        x_se_129 = torch.conv2d(
            x_se_128,
            l_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_128 = l_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_130 = torch.nn.functional.silu(x_se_129, inplace=True)
        x_se_129 = None
        x_se_131 = torch.conv2d(
            x_se_130,
            l_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_130 = l_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_32 = torch.sigmoid(x_se_131)
        x_se_131 = None
        x_314 = x_313 * sigmoid_32
        x_313 = sigmoid_32 = None
        x_315 = torch.conv2d(
            x_314,
            l_self_modules_blocks_modules_5_modules_5_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_314 = l_self_modules_blocks_modules_5_modules_5_modules_conv_pwl_parameters_weight_ = (None)
        x_316 = torch.nn.functional.batch_norm(
            x_315,
            l_self_modules_blocks_modules_5_modules_5_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_5_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_5_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_5_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_315 = (
            l_self_modules_blocks_modules_5_modules_5_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_5_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_5_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_5_modules_bn3_parameters_bias_
        ) = None
        x_317 = x_316 + x_307
        x_316 = x_307 = None
        x_318 = torch.conv2d(
            x_317,
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
        x_319 = torch.nn.functional.batch_norm(
            x_318,
            l_self_modules_blocks_modules_5_modules_6_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_6_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_6_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_6_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_318 = (
            l_self_modules_blocks_modules_5_modules_6_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_6_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_6_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_6_modules_bn1_parameters_bias_
        ) = None
        x_320 = torch.nn.functional.silu(x_319, inplace=True)
        x_319 = None
        x_321 = torch.conv2d(
            x_320,
            l_self_modules_blocks_modules_5_modules_6_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1824,
        )
        x_320 = (
            l_self_modules_blocks_modules_5_modules_6_modules_conv_dw_parameters_weight_
        ) = None
        x_322 = torch.nn.functional.batch_norm(
            x_321,
            l_self_modules_blocks_modules_5_modules_6_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_6_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_6_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_6_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_321 = (
            l_self_modules_blocks_modules_5_modules_6_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_6_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_6_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_6_modules_bn2_parameters_bias_
        ) = None
        x_323 = torch.nn.functional.silu(x_322, inplace=True)
        x_322 = None
        x_se_132 = x_323.mean((2, 3), keepdim=True)
        x_se_133 = torch.conv2d(
            x_se_132,
            l_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_132 = l_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_134 = torch.nn.functional.silu(x_se_133, inplace=True)
        x_se_133 = None
        x_se_135 = torch.conv2d(
            x_se_134,
            l_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_134 = l_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_33 = torch.sigmoid(x_se_135)
        x_se_135 = None
        x_324 = x_323 * sigmoid_33
        x_323 = sigmoid_33 = None
        x_325 = torch.conv2d(
            x_324,
            l_self_modules_blocks_modules_5_modules_6_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_324 = l_self_modules_blocks_modules_5_modules_6_modules_conv_pwl_parameters_weight_ = (None)
        x_326 = torch.nn.functional.batch_norm(
            x_325,
            l_self_modules_blocks_modules_5_modules_6_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_6_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_6_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_6_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_325 = (
            l_self_modules_blocks_modules_5_modules_6_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_6_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_6_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_6_modules_bn3_parameters_bias_
        ) = None
        x_327 = x_326 + x_317
        x_326 = x_317 = None
        x_328 = torch.conv2d(
            x_327,
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
        x_329 = torch.nn.functional.batch_norm(
            x_328,
            l_self_modules_blocks_modules_5_modules_7_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_7_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_7_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_7_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_328 = (
            l_self_modules_blocks_modules_5_modules_7_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_7_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_7_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_7_modules_bn1_parameters_bias_
        ) = None
        x_330 = torch.nn.functional.silu(x_329, inplace=True)
        x_329 = None
        x_331 = torch.conv2d(
            x_330,
            l_self_modules_blocks_modules_5_modules_7_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1824,
        )
        x_330 = (
            l_self_modules_blocks_modules_5_modules_7_modules_conv_dw_parameters_weight_
        ) = None
        x_332 = torch.nn.functional.batch_norm(
            x_331,
            l_self_modules_blocks_modules_5_modules_7_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_7_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_7_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_7_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_331 = (
            l_self_modules_blocks_modules_5_modules_7_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_7_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_7_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_7_modules_bn2_parameters_bias_
        ) = None
        x_333 = torch.nn.functional.silu(x_332, inplace=True)
        x_332 = None
        x_se_136 = x_333.mean((2, 3), keepdim=True)
        x_se_137 = torch.conv2d(
            x_se_136,
            l_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_136 = l_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_138 = torch.nn.functional.silu(x_se_137, inplace=True)
        x_se_137 = None
        x_se_139 = torch.conv2d(
            x_se_138,
            l_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_138 = l_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_34 = torch.sigmoid(x_se_139)
        x_se_139 = None
        x_334 = x_333 * sigmoid_34
        x_333 = sigmoid_34 = None
        x_335 = torch.conv2d(
            x_334,
            l_self_modules_blocks_modules_5_modules_7_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_334 = l_self_modules_blocks_modules_5_modules_7_modules_conv_pwl_parameters_weight_ = (None)
        x_336 = torch.nn.functional.batch_norm(
            x_335,
            l_self_modules_blocks_modules_5_modules_7_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_7_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_7_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_7_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_335 = (
            l_self_modules_blocks_modules_5_modules_7_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_7_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_7_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_7_modules_bn3_parameters_bias_
        ) = None
        x_337 = x_336 + x_327
        x_336 = x_327 = None
        x_338 = torch.conv2d(
            x_337,
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
        x_339 = torch.nn.functional.batch_norm(
            x_338,
            l_self_modules_blocks_modules_5_modules_8_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_8_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_8_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_8_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_338 = (
            l_self_modules_blocks_modules_5_modules_8_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_8_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_8_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_8_modules_bn1_parameters_bias_
        ) = None
        x_340 = torch.nn.functional.silu(x_339, inplace=True)
        x_339 = None
        x_341 = torch.conv2d(
            x_340,
            l_self_modules_blocks_modules_5_modules_8_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1824,
        )
        x_340 = (
            l_self_modules_blocks_modules_5_modules_8_modules_conv_dw_parameters_weight_
        ) = None
        x_342 = torch.nn.functional.batch_norm(
            x_341,
            l_self_modules_blocks_modules_5_modules_8_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_8_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_8_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_8_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_341 = (
            l_self_modules_blocks_modules_5_modules_8_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_8_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_8_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_8_modules_bn2_parameters_bias_
        ) = None
        x_343 = torch.nn.functional.silu(x_342, inplace=True)
        x_342 = None
        x_se_140 = x_343.mean((2, 3), keepdim=True)
        x_se_141 = torch.conv2d(
            x_se_140,
            l_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_140 = l_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_142 = torch.nn.functional.silu(x_se_141, inplace=True)
        x_se_141 = None
        x_se_143 = torch.conv2d(
            x_se_142,
            l_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_142 = l_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_35 = torch.sigmoid(x_se_143)
        x_se_143 = None
        x_344 = x_343 * sigmoid_35
        x_343 = sigmoid_35 = None
        x_345 = torch.conv2d(
            x_344,
            l_self_modules_blocks_modules_5_modules_8_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_344 = l_self_modules_blocks_modules_5_modules_8_modules_conv_pwl_parameters_weight_ = (None)
        x_346 = torch.nn.functional.batch_norm(
            x_345,
            l_self_modules_blocks_modules_5_modules_8_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_8_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_8_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_8_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_345 = (
            l_self_modules_blocks_modules_5_modules_8_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_8_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_8_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_8_modules_bn3_parameters_bias_
        ) = None
        x_347 = x_346 + x_337
        x_346 = x_337 = None
        x_348 = torch.conv2d(
            x_347,
            l_self_modules_blocks_modules_6_modules_0_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_347 = (
            l_self_modules_blocks_modules_6_modules_0_modules_conv_pw_parameters_weight_
        ) = None
        x_349 = torch.nn.functional.batch_norm(
            x_348,
            l_self_modules_blocks_modules_6_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_348 = (
            l_self_modules_blocks_modules_6_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_6_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_6_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_350 = torch.nn.functional.silu(x_349, inplace=True)
        x_349 = None
        x_351 = torch.conv2d(
            x_350,
            l_self_modules_blocks_modules_6_modules_0_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1824,
        )
        x_350 = (
            l_self_modules_blocks_modules_6_modules_0_modules_conv_dw_parameters_weight_
        ) = None
        x_352 = torch.nn.functional.batch_norm(
            x_351,
            l_self_modules_blocks_modules_6_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_351 = (
            l_self_modules_blocks_modules_6_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_6_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_6_modules_0_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_0_modules_bn2_parameters_bias_
        ) = None
        x_353 = torch.nn.functional.silu(x_352, inplace=True)
        x_352 = None
        x_se_144 = x_353.mean((2, 3), keepdim=True)
        x_se_145 = torch.conv2d(
            x_se_144,
            l_self_modules_blocks_modules_6_modules_0_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_0_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_144 = l_self_modules_blocks_modules_6_modules_0_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_6_modules_0_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_146 = torch.nn.functional.silu(x_se_145, inplace=True)
        x_se_145 = None
        x_se_147 = torch.conv2d(
            x_se_146,
            l_self_modules_blocks_modules_6_modules_0_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_0_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_146 = l_self_modules_blocks_modules_6_modules_0_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_6_modules_0_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_36 = torch.sigmoid(x_se_147)
        x_se_147 = None
        x_354 = x_353 * sigmoid_36
        x_353 = sigmoid_36 = None
        x_355 = torch.conv2d(
            x_354,
            l_self_modules_blocks_modules_6_modules_0_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_354 = l_self_modules_blocks_modules_6_modules_0_modules_conv_pwl_parameters_weight_ = (None)
        x_356 = torch.nn.functional.batch_norm(
            x_355,
            l_self_modules_blocks_modules_6_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_355 = (
            l_self_modules_blocks_modules_6_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_6_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_6_modules_0_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_0_modules_bn3_parameters_bias_
        ) = None
        x_357 = torch.conv2d(
            x_356,
            l_self_modules_blocks_modules_6_modules_1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_6_modules_1_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_358 = torch.nn.functional.batch_norm(
            x_357,
            l_self_modules_blocks_modules_6_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_357 = (
            l_self_modules_blocks_modules_6_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_6_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_6_modules_1_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_1_modules_bn1_parameters_bias_
        ) = None
        x_359 = torch.nn.functional.silu(x_358, inplace=True)
        x_358 = None
        x_360 = torch.conv2d(
            x_359,
            l_self_modules_blocks_modules_6_modules_1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            3072,
        )
        x_359 = (
            l_self_modules_blocks_modules_6_modules_1_modules_conv_dw_parameters_weight_
        ) = None
        x_361 = torch.nn.functional.batch_norm(
            x_360,
            l_self_modules_blocks_modules_6_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_360 = (
            l_self_modules_blocks_modules_6_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_6_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_6_modules_1_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_1_modules_bn2_parameters_bias_
        ) = None
        x_362 = torch.nn.functional.silu(x_361, inplace=True)
        x_361 = None
        x_se_148 = x_362.mean((2, 3), keepdim=True)
        x_se_149 = torch.conv2d(
            x_se_148,
            l_self_modules_blocks_modules_6_modules_1_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_1_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_148 = l_self_modules_blocks_modules_6_modules_1_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_6_modules_1_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_150 = torch.nn.functional.silu(x_se_149, inplace=True)
        x_se_149 = None
        x_se_151 = torch.conv2d(
            x_se_150,
            l_self_modules_blocks_modules_6_modules_1_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_1_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_150 = l_self_modules_blocks_modules_6_modules_1_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_6_modules_1_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_37 = torch.sigmoid(x_se_151)
        x_se_151 = None
        x_363 = x_362 * sigmoid_37
        x_362 = sigmoid_37 = None
        x_364 = torch.conv2d(
            x_363,
            l_self_modules_blocks_modules_6_modules_1_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_363 = l_self_modules_blocks_modules_6_modules_1_modules_conv_pwl_parameters_weight_ = (None)
        x_365 = torch.nn.functional.batch_norm(
            x_364,
            l_self_modules_blocks_modules_6_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_364 = (
            l_self_modules_blocks_modules_6_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_6_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_6_modules_1_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_1_modules_bn3_parameters_bias_
        ) = None
        x_366 = x_365 + x_356
        x_365 = x_356 = None
        x_367 = torch.conv2d(
            x_366,
            l_self_modules_blocks_modules_6_modules_2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_6_modules_2_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_368 = torch.nn.functional.batch_norm(
            x_367,
            l_self_modules_blocks_modules_6_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_367 = (
            l_self_modules_blocks_modules_6_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_6_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_6_modules_2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_2_modules_bn1_parameters_bias_
        ) = None
        x_369 = torch.nn.functional.silu(x_368, inplace=True)
        x_368 = None
        x_370 = torch.conv2d(
            x_369,
            l_self_modules_blocks_modules_6_modules_2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            3072,
        )
        x_369 = (
            l_self_modules_blocks_modules_6_modules_2_modules_conv_dw_parameters_weight_
        ) = None
        x_371 = torch.nn.functional.batch_norm(
            x_370,
            l_self_modules_blocks_modules_6_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_370 = (
            l_self_modules_blocks_modules_6_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_6_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_6_modules_2_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_2_modules_bn2_parameters_bias_
        ) = None
        x_372 = torch.nn.functional.silu(x_371, inplace=True)
        x_371 = None
        x_se_152 = x_372.mean((2, 3), keepdim=True)
        x_se_153 = torch.conv2d(
            x_se_152,
            l_self_modules_blocks_modules_6_modules_2_modules_se_modules_conv_reduce_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_2_modules_se_modules_conv_reduce_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_152 = l_self_modules_blocks_modules_6_modules_2_modules_se_modules_conv_reduce_parameters_weight_ = l_self_modules_blocks_modules_6_modules_2_modules_se_modules_conv_reduce_parameters_bias_ = (None)
        x_se_154 = torch.nn.functional.silu(x_se_153, inplace=True)
        x_se_153 = None
        x_se_155 = torch.conv2d(
            x_se_154,
            l_self_modules_blocks_modules_6_modules_2_modules_se_modules_conv_expand_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_2_modules_se_modules_conv_expand_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_154 = l_self_modules_blocks_modules_6_modules_2_modules_se_modules_conv_expand_parameters_weight_ = l_self_modules_blocks_modules_6_modules_2_modules_se_modules_conv_expand_parameters_bias_ = (None)
        sigmoid_38 = torch.sigmoid(x_se_155)
        x_se_155 = None
        x_373 = x_372 * sigmoid_38
        x_372 = sigmoid_38 = None
        x_374 = torch.conv2d(
            x_373,
            l_self_modules_blocks_modules_6_modules_2_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_373 = l_self_modules_blocks_modules_6_modules_2_modules_conv_pwl_parameters_weight_ = (None)
        x_375 = torch.nn.functional.batch_norm(
            x_374,
            l_self_modules_blocks_modules_6_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_374 = (
            l_self_modules_blocks_modules_6_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_6_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_6_modules_2_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_2_modules_bn3_parameters_bias_
        ) = None
        x_376 = x_375 + x_366
        x_375 = x_366 = None
        x_377 = torch.conv2d(
            x_376,
            l_self_modules_conv_head_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_376 = l_self_modules_conv_head_parameters_weight_ = None
        x_378 = torch.nn.functional.batch_norm(
            x_377,
            l_self_modules_bn2_buffers_running_mean_,
            l_self_modules_bn2_buffers_running_var_,
            l_self_modules_bn2_parameters_weight_,
            l_self_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_377 = (
            l_self_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_bn2_parameters_weight_
        ) = l_self_modules_bn2_parameters_bias_ = None
        x_379 = torch.nn.functional.silu(x_378, inplace=True)
        x_378 = None
        x_380 = torch.nn.functional.adaptive_avg_pool2d(x_379, 1)
        x_379 = None
        x_381 = x_380.flatten(1, -1)
        x_380 = None
        x_382 = torch._C._nn.linear(
            x_381,
            l_self_modules_classifier_parameters_weight_,
            l_self_modules_classifier_parameters_bias_,
        )
        x_381 = (
            l_self_modules_classifier_parameters_weight_
        ) = l_self_modules_classifier_parameters_bias_ = None
        return (x_382,)
