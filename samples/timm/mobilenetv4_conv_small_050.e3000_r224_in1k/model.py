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
        L_self_modules_blocks_modules_1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_4_modules_dw_mid_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_4_modules_dw_mid_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_4_modules_dw_mid_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_4_modules_dw_mid_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_4_modules_dw_mid_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_conv_head_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_norm_head_buffers_running_mean_: torch.Tensor,
        L_self_modules_norm_head_buffers_running_var_: torch.Tensor,
        L_self_modules_norm_head_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_norm_head_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_blocks_modules_1_modules_0_modules_conv_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_0_modules_conv_parameters_weight_
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
        l_self_modules_blocks_modules_1_modules_1_modules_conv_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_1_modules_conv_parameters_weight_
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
        l_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_4_modules_dw_mid_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_4_modules_dw_mid_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_4_modules_dw_mid_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_4_modules_dw_mid_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_4_modules_dw_mid_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_4_modules_dw_mid_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_4_modules_dw_mid_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_4_modules_dw_mid_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_4_modules_dw_mid_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_4_modules_dw_mid_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_4_modules_0_modules_conv_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_0_modules_conv_parameters_weight_
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
        l_self_modules_conv_head_parameters_weight_ = (
            L_self_modules_conv_head_parameters_weight_
        )
        l_self_modules_norm_head_buffers_running_mean_ = (
            L_self_modules_norm_head_buffers_running_mean_
        )
        l_self_modules_norm_head_buffers_running_var_ = (
            L_self_modules_norm_head_buffers_running_var_
        )
        l_self_modules_norm_head_parameters_weight_ = (
            L_self_modules_norm_head_parameters_weight_
        )
        l_self_modules_norm_head_parameters_bias_ = (
            L_self_modules_norm_head_parameters_bias_
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
        x_3 = torch.conv2d(
            x_2,
            l_self_modules_blocks_modules_0_modules_0_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_2 = (
            l_self_modules_blocks_modules_0_modules_0_modules_conv_parameters_weight_
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
        x_5 = torch.nn.functional.relu(x_4, inplace=True)
        x_4 = None
        x_6 = torch.conv2d(
            x_5,
            l_self_modules_blocks_modules_0_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_5 = (
            l_self_modules_blocks_modules_0_modules_1_modules_conv_parameters_weight_
        ) = None
        x_7 = torch.nn.functional.batch_norm(
            x_6,
            l_self_modules_blocks_modules_0_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_0_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_0_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_6 = (
            l_self_modules_blocks_modules_0_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_0_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_0_modules_1_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_1_modules_bn1_parameters_bias_
        ) = None
        x_8 = torch.nn.functional.relu(x_7, inplace=True)
        x_7 = None
        x_9 = torch.conv2d(
            x_8,
            l_self_modules_blocks_modules_1_modules_0_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_8 = (
            l_self_modules_blocks_modules_1_modules_0_modules_conv_parameters_weight_
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
        x_11 = torch.nn.functional.relu(x_10, inplace=True)
        x_10 = None
        x_12 = torch.conv2d(
            x_11,
            l_self_modules_blocks_modules_1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_11 = (
            l_self_modules_blocks_modules_1_modules_1_modules_conv_parameters_weight_
        ) = None
        x_13 = torch.nn.functional.batch_norm(
            x_12,
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_12 = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_bias_
        ) = None
        x_14 = torch.nn.functional.relu(x_13, inplace=True)
        x_13 = None
        x_15 = torch.conv2d(
            x_14,
            l_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            32,
        )
        x_14 = l_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_conv_parameters_weight_ = (None)
        x_16 = torch.nn.functional.batch_norm(
            x_15,
            l_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_15 = l_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_17 = torch.conv2d(
            x_16,
            l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_16 = l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_18 = torch.nn.functional.batch_norm(
            x_17,
            l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_17 = l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_19 = torch.nn.functional.relu(x_18, inplace=True)
        x_18 = None
        x_20 = torch.conv2d(
            x_19,
            l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (2, 2),
            (1, 1),
            96,
        )
        x_19 = l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_21 = torch.nn.functional.batch_norm(
            x_20,
            l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_20 = l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_22 = torch.nn.functional.relu(x_21, inplace=True)
        x_21 = None
        x_23 = torch.conv2d(
            x_22,
            l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_22 = l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_24 = torch.nn.functional.batch_norm(
            x_23,
            l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_23 = l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_25 = torch.conv2d(
            x_24,
            l_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_conv_parameters_weight_ = (
            None
        )
        x_26 = torch.nn.functional.batch_norm(
            x_25,
            l_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_25 = l_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_27 = torch.nn.functional.relu(x_26, inplace=True)
        x_26 = None
        x_28 = torch.conv2d(
            x_27,
            l_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            96,
        )
        x_27 = l_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_29 = torch.nn.functional.batch_norm(
            x_28,
            l_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_28 = l_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_30 = torch.nn.functional.relu(x_29, inplace=True)
        x_29 = None
        x_31 = torch.conv2d(
            x_30,
            l_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_30 = l_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_32 = torch.nn.functional.batch_norm(
            x_31,
            l_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_31 = l_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_33 = x_32 + x_24
        x_32 = x_24 = None
        x_34 = torch.conv2d(
            x_33,
            l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_conv_parameters_weight_ = (
            None
        )
        x_35 = torch.nn.functional.batch_norm(
            x_34,
            l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_34 = l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_36 = torch.nn.functional.relu(x_35, inplace=True)
        x_35 = None
        x_37 = torch.conv2d(
            x_36,
            l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            96,
        )
        x_36 = l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_38 = torch.nn.functional.batch_norm(
            x_37,
            l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_37 = l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_39 = torch.nn.functional.relu(x_38, inplace=True)
        x_38 = None
        x_40 = torch.conv2d(
            x_39,
            l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_39 = l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_41 = torch.nn.functional.batch_norm(
            x_40,
            l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_40 = l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_42 = x_41 + x_33
        x_41 = x_33 = None
        x_43 = torch.conv2d(
            x_42,
            l_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_conv_parameters_weight_ = (
            None
        )
        x_44 = torch.nn.functional.batch_norm(
            x_43,
            l_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_43 = l_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_45 = torch.nn.functional.relu(x_44, inplace=True)
        x_44 = None
        x_46 = torch.conv2d(
            x_45,
            l_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            96,
        )
        x_45 = l_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_47 = torch.nn.functional.batch_norm(
            x_46,
            l_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_46 = l_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_48 = torch.nn.functional.relu(x_47, inplace=True)
        x_47 = None
        x_49 = torch.conv2d(
            x_48,
            l_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_48 = l_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_50 = torch.nn.functional.batch_norm(
            x_49,
            l_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_49 = l_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_51 = x_50 + x_42
        x_50 = x_42 = None
        x_52 = torch.conv2d(
            x_51,
            l_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_conv_parameters_weight_ = (
            None
        )
        x_53 = torch.nn.functional.batch_norm(
            x_52,
            l_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_52 = l_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_54 = torch.nn.functional.relu(x_53, inplace=True)
        x_53 = None
        x_55 = torch.conv2d(
            x_54,
            l_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            96,
        )
        x_54 = l_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_56 = torch.nn.functional.batch_norm(
            x_55,
            l_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_55 = l_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_57 = torch.nn.functional.relu(x_56, inplace=True)
        x_56 = None
        x_58 = torch.conv2d(
            x_57,
            l_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_57 = l_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_59 = torch.nn.functional.batch_norm(
            x_58,
            l_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_58 = l_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_60 = x_59 + x_51
        x_59 = x_51 = None
        x_61 = torch.conv2d(
            x_60,
            l_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            48,
        )
        l_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_conv_parameters_weight_ = (
            None
        )
        x_62 = torch.nn.functional.batch_norm(
            x_61,
            l_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_61 = l_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_63 = torch.conv2d(
            x_62,
            l_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_62 = l_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_64 = torch.nn.functional.batch_norm(
            x_63,
            l_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_63 = l_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_65 = torch.nn.functional.relu(x_64, inplace=True)
        x_64 = None
        x_66 = torch.conv2d(
            x_65,
            l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_65 = l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_67 = torch.nn.functional.batch_norm(
            x_66,
            l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_66 = l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_68 = x_67 + x_60
        x_67 = x_60 = None
        x_69 = torch.conv2d(
            x_68,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            48,
        )
        x_68 = l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_conv_parameters_weight_ = (None)
        x_70 = torch.nn.functional.batch_norm(
            x_69,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_69 = l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_71 = torch.conv2d(
            x_70,
            l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_70 = l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_72 = torch.nn.functional.batch_norm(
            x_71,
            l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_71 = l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_73 = torch.nn.functional.relu(x_72, inplace=True)
        x_72 = None
        x_74 = torch.conv2d(
            x_73,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            288,
        )
        x_73 = l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_75 = torch.nn.functional.batch_norm(
            x_74,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_74 = l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_76 = torch.nn.functional.relu(x_75, inplace=True)
        x_75 = None
        x_77 = torch.conv2d(
            x_76,
            l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_76 = l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_78 = torch.nn.functional.batch_norm(
            x_77,
            l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_77 = l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_79 = torch.conv2d(
            x_78,
            l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            64,
        )
        l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_conv_parameters_weight_ = (
            None
        )
        x_80 = torch.nn.functional.batch_norm(
            x_79,
            l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_79 = l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_81 = torch.conv2d(
            x_80,
            l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_80 = l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_82 = torch.nn.functional.batch_norm(
            x_81,
            l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_81 = l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_83 = torch.nn.functional.relu(x_82, inplace=True)
        x_82 = None
        x_84 = torch.conv2d(
            x_83,
            l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            256,
        )
        x_83 = l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_85 = torch.nn.functional.batch_norm(
            x_84,
            l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_84 = l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_86 = torch.nn.functional.relu(x_85, inplace=True)
        x_85 = None
        x_87 = torch.conv2d(
            x_86,
            l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_86 = l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_88 = torch.nn.functional.batch_norm(
            x_87,
            l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_87 = l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_89 = x_88 + x_78
        x_88 = x_78 = None
        x_90 = torch.conv2d(
            x_89,
            l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_conv_parameters_weight_ = (
            None
        )
        x_91 = torch.nn.functional.batch_norm(
            x_90,
            l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_90 = l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_92 = torch.nn.functional.relu(x_91, inplace=True)
        x_91 = None
        x_93 = torch.conv2d(
            x_92,
            l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            256,
        )
        x_92 = l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_94 = torch.nn.functional.batch_norm(
            x_93,
            l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_93 = l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_95 = torch.nn.functional.relu(x_94, inplace=True)
        x_94 = None
        x_96 = torch.conv2d(
            x_95,
            l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_95 = l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_97 = torch.nn.functional.batch_norm(
            x_96,
            l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_96 = l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_98 = x_97 + x_89
        x_97 = x_89 = None
        x_99 = torch.conv2d(
            x_98,
            l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_conv_parameters_weight_ = (
            None
        )
        x_100 = torch.nn.functional.batch_norm(
            x_99,
            l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_99 = l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_101 = torch.nn.functional.relu(x_100, inplace=True)
        x_100 = None
        x_102 = torch.conv2d(
            x_101,
            l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            192,
        )
        x_101 = l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_103 = torch.nn.functional.batch_norm(
            x_102,
            l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_102 = l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_104 = torch.nn.functional.relu(x_103, inplace=True)
        x_103 = None
        x_105 = torch.conv2d(
            x_104,
            l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_104 = l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_106 = torch.nn.functional.batch_norm(
            x_105,
            l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_105 = l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_107 = x_106 + x_98
        x_106 = x_98 = None
        x_108 = torch.conv2d(
            x_107,
            l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_conv_parameters_weight_ = (
            None
        )
        x_109 = torch.nn.functional.batch_norm(
            x_108,
            l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_108 = l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_110 = torch.nn.functional.relu(x_109, inplace=True)
        x_109 = None
        x_111 = torch.conv2d(
            x_110,
            l_self_modules_blocks_modules_3_modules_4_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        x_110 = l_self_modules_blocks_modules_3_modules_4_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_112 = torch.nn.functional.batch_norm(
            x_111,
            l_self_modules_blocks_modules_3_modules_4_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_4_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_4_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_4_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_111 = l_self_modules_blocks_modules_3_modules_4_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_4_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_4_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_4_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_113 = torch.nn.functional.relu(x_112, inplace=True)
        x_112 = None
        x_114 = torch.conv2d(
            x_113,
            l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_113 = l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_115 = torch.nn.functional.batch_norm(
            x_114,
            l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_114 = l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_116 = x_115 + x_107
        x_115 = x_107 = None
        x_117 = torch.conv2d(
            x_116,
            l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_conv_parameters_weight_ = (
            None
        )
        x_118 = torch.nn.functional.batch_norm(
            x_117,
            l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_117 = l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_119 = torch.nn.functional.relu(x_118, inplace=True)
        x_118 = None
        x_120 = torch.conv2d(
            x_119,
            l_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        x_119 = l_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_121 = torch.nn.functional.batch_norm(
            x_120,
            l_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_120 = l_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_122 = torch.nn.functional.relu(x_121, inplace=True)
        x_121 = None
        x_123 = torch.conv2d(
            x_122,
            l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_122 = l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_124 = torch.nn.functional.batch_norm(
            x_123,
            l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_123 = l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_125 = x_124 + x_116
        x_124 = x_116 = None
        x_126 = torch.conv2d(
            x_125,
            l_self_modules_blocks_modules_4_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_125 = (
            l_self_modules_blocks_modules_4_modules_0_modules_conv_parameters_weight_
        ) = None
        x_127 = torch.nn.functional.batch_norm(
            x_126,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_126 = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_128 = torch.nn.functional.relu(x_127, inplace=True)
        x_127 = None
        x_129 = torch.nn.functional.adaptive_avg_pool2d(x_128, 1)
        x_128 = None
        x_130 = torch.conv2d(
            x_129,
            l_self_modules_conv_head_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_129 = l_self_modules_conv_head_parameters_weight_ = None
        x_131 = torch.nn.functional.batch_norm(
            x_130,
            l_self_modules_norm_head_buffers_running_mean_,
            l_self_modules_norm_head_buffers_running_var_,
            l_self_modules_norm_head_parameters_weight_,
            l_self_modules_norm_head_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_130 = (
            l_self_modules_norm_head_buffers_running_mean_
        ) = (
            l_self_modules_norm_head_buffers_running_var_
        ) = (
            l_self_modules_norm_head_parameters_weight_
        ) = l_self_modules_norm_head_parameters_bias_ = None
        x_132 = torch.nn.functional.relu(x_131, inplace=True)
        x_131 = None
        x_133 = x_132.flatten(1, -1)
        x_132 = None
        x_134 = torch._C._nn.linear(
            x_133,
            l_self_modules_classifier_parameters_weight_,
            l_self_modules_classifier_parameters_bias_,
        )
        x_133 = (
            l_self_modules_classifier_parameters_weight_
        ) = l_self_modules_classifier_parameters_bias_ = None
        return (x_134,)
