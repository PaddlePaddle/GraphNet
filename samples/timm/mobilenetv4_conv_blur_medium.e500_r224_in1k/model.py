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
        L_self_modules_blocks_modules_0_modules_0_modules_aa_buffers_filt_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_0_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_aa_buffers_filt_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_aa_buffers_filt_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_aa_buffers_filt_: torch.Tensor,
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
        L_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_6_modules_dw_mid_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_6_modules_dw_mid_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_6_modules_dw_mid_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_6_modules_dw_mid_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_6_modules_dw_mid_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_7_modules_dw_mid_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_7_modules_dw_mid_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_7_modules_dw_mid_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_7_modules_dw_mid_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_7_modules_dw_mid_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_blocks_modules_0_modules_0_modules_aa_buffers_filt_ = (
            L_self_modules_blocks_modules_0_modules_0_modules_aa_buffers_filt_
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
        l_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_aa_buffers_filt_ = L_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_aa_buffers_filt_
        l_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_bn_parameters_bias_
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
        l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_aa_buffers_filt_ = L_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_aa_buffers_filt_
        l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_bn_parameters_bias_
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
        l_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_bn_parameters_bias_
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
        l_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_bn_parameters_bias_
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
        l_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_bn_parameters_bias_
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
        l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_parameters_bias_
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
        l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_aa_buffers_filt_ = L_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_aa_buffers_filt_
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
        l_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_parameters_bias_
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
        l_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_parameters_bias_
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
        l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_6_modules_dw_mid_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_6_modules_dw_mid_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_6_modules_dw_mid_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_6_modules_dw_mid_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_6_modules_dw_mid_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_6_modules_dw_mid_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_6_modules_dw_mid_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_6_modules_dw_mid_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_6_modules_dw_mid_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_6_modules_dw_mid_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_7_modules_dw_mid_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_7_modules_dw_mid_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_7_modules_dw_mid_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_7_modules_dw_mid_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_7_modules_dw_mid_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_7_modules_dw_mid_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_7_modules_dw_mid_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_7_modules_dw_mid_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_7_modules_dw_mid_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_7_modules_dw_mid_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_parameters_bias_
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
            l_self_modules_blocks_modules_0_modules_0_modules_conv_exp_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_2 = l_self_modules_blocks_modules_0_modules_0_modules_conv_exp_parameters_weight_ = (None)
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
        x_6 = torch._C._nn.pad(x_5, [1, 1, 1, 1], "constant", None)
        x_5 = None
        x_7 = torch.conv2d(
            x_6,
            l_self_modules_blocks_modules_0_modules_0_modules_aa_buffers_filt_,
            stride=2,
            groups=128,
        )
        x_6 = l_self_modules_blocks_modules_0_modules_0_modules_aa_buffers_filt_ = None
        x_8 = torch.conv2d(
            x_7,
            l_self_modules_blocks_modules_0_modules_0_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_7 = l_self_modules_blocks_modules_0_modules_0_modules_conv_pwl_parameters_weight_ = (None)
        x_9 = torch.nn.functional.batch_norm(
            x_8,
            l_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_8 = (
            l_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_bias_
        ) = None
        x_10 = torch.conv2d(
            x_9,
            l_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            48,
        )
        x_9 = l_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_conv_parameters_weight_ = (None)
        x_11 = torch.nn.functional.batch_norm(
            x_10,
            l_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_10 = l_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_12 = torch.conv2d(
            x_11,
            l_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_11 = l_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_13 = torch.nn.functional.batch_norm(
            x_12,
            l_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_12 = l_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_14 = torch.nn.functional.relu(x_13, inplace=True)
        x_13 = None
        x_15 = torch.conv2d(
            x_14,
            l_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            192,
        )
        x_14 = l_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_16 = torch.nn.functional.batch_norm(
            x_15,
            l_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_15 = l_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_17 = torch.nn.functional.relu(x_16, inplace=True)
        x_16 = None
        x_18 = torch._C._nn.pad(x_17, [1, 1, 1, 1], "constant", None)
        x_17 = None
        x_19 = torch.conv2d(
            x_18,
            l_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_aa_buffers_filt_,
            stride=2,
            groups=192,
        )
        x_18 = l_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_aa_buffers_filt_ = (None)
        x_20 = torch.conv2d(
            x_19,
            l_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_19 = l_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_21 = torch.nn.functional.batch_norm(
            x_20,
            l_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_20 = l_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_22 = torch.conv2d(
            x_21,
            l_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            80,
        )
        l_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_conv_parameters_weight_ = (
            None
        )
        x_23 = torch.nn.functional.batch_norm(
            x_22,
            l_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_22 = l_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_24 = torch.conv2d(
            x_23,
            l_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_23 = l_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_25 = torch.nn.functional.batch_norm(
            x_24,
            l_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_24 = l_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_26 = torch.nn.functional.relu(x_25, inplace=True)
        x_25 = None
        x_27 = torch.conv2d(
            x_26,
            l_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            160,
        )
        x_26 = l_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_28 = torch.nn.functional.batch_norm(
            x_27,
            l_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_27 = l_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_29 = torch.nn.functional.relu(x_28, inplace=True)
        x_28 = None
        x_30 = torch.conv2d(
            x_29,
            l_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_29 = l_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_31 = torch.nn.functional.batch_norm(
            x_30,
            l_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_30 = l_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_32 = x_31 + x_21
        x_31 = x_21 = None
        x_33 = torch.conv2d(
            x_32,
            l_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            80,
        )
        x_32 = l_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_conv_parameters_weight_ = (None)
        x_34 = torch.nn.functional.batch_norm(
            x_33,
            l_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_33 = l_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_35 = torch.conv2d(
            x_34,
            l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_34 = l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_36 = torch.nn.functional.batch_norm(
            x_35,
            l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_35 = l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_37 = torch.nn.functional.relu(x_36, inplace=True)
        x_36 = None
        x_38 = torch.conv2d(
            x_37,
            l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            480,
        )
        x_37 = l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_39 = torch.nn.functional.batch_norm(
            x_38,
            l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_38 = l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_40 = torch.nn.functional.relu(x_39, inplace=True)
        x_39 = None
        x_41 = torch._C._nn.pad(x_40, [1, 1, 1, 1], "constant", None)
        x_40 = None
        x_42 = torch.conv2d(
            x_41,
            l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_aa_buffers_filt_,
            stride=2,
            groups=480,
        )
        x_41 = l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_aa_buffers_filt_ = (None)
        x_43 = torch.conv2d(
            x_42,
            l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_42 = l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_44 = torch.nn.functional.batch_norm(
            x_43,
            l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_43 = l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_45 = torch.conv2d(
            x_44,
            l_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            160,
        )
        l_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_conv_parameters_weight_ = (
            None
        )
        x_46 = torch.nn.functional.batch_norm(
            x_45,
            l_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_45 = l_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_47 = torch.conv2d(
            x_46,
            l_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_46 = l_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_48 = torch.nn.functional.batch_norm(
            x_47,
            l_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_47 = l_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_49 = torch.nn.functional.relu(x_48, inplace=True)
        x_48 = None
        x_50 = torch.conv2d(
            x_49,
            l_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            640,
        )
        x_49 = l_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_51 = torch.nn.functional.batch_norm(
            x_50,
            l_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_50 = l_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_52 = torch.nn.functional.relu(x_51, inplace=True)
        x_51 = None
        x_53 = torch.conv2d(
            x_52,
            l_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_52 = l_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_54 = torch.nn.functional.batch_norm(
            x_53,
            l_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_53 = l_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_55 = x_54 + x_44
        x_54 = x_44 = None
        x_56 = torch.conv2d(
            x_55,
            l_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            160,
        )
        l_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_conv_parameters_weight_ = (
            None
        )
        x_57 = torch.nn.functional.batch_norm(
            x_56,
            l_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_56 = l_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_58 = torch.conv2d(
            x_57,
            l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_57 = l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_59 = torch.nn.functional.batch_norm(
            x_58,
            l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_58 = l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_60 = torch.nn.functional.relu(x_59, inplace=True)
        x_59 = None
        x_61 = torch.conv2d(
            x_60,
            l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            640,
        )
        x_60 = l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_62 = torch.nn.functional.batch_norm(
            x_61,
            l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_61 = l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_63 = torch.nn.functional.relu(x_62, inplace=True)
        x_62 = None
        x_64 = torch.conv2d(
            x_63,
            l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_63 = l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_65 = torch.nn.functional.batch_norm(
            x_64,
            l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_64 = l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_66 = x_65 + x_55
        x_65 = x_55 = None
        x_67 = torch.conv2d(
            x_66,
            l_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            160,
        )
        l_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_conv_parameters_weight_ = (
            None
        )
        x_68 = torch.nn.functional.batch_norm(
            x_67,
            l_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_67 = l_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_69 = torch.conv2d(
            x_68,
            l_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_68 = l_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_70 = torch.nn.functional.batch_norm(
            x_69,
            l_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_69 = l_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_71 = torch.nn.functional.relu(x_70, inplace=True)
        x_70 = None
        x_72 = torch.conv2d(
            x_71,
            l_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            640,
        )
        x_71 = l_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_73 = torch.nn.functional.batch_norm(
            x_72,
            l_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_72 = l_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_74 = torch.nn.functional.relu(x_73, inplace=True)
        x_73 = None
        x_75 = torch.conv2d(
            x_74,
            l_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_74 = l_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_76 = torch.nn.functional.batch_norm(
            x_75,
            l_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_75 = l_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_77 = x_76 + x_66
        x_76 = x_66 = None
        x_78 = torch.conv2d(
            x_77,
            l_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            160,
        )
        l_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_conv_parameters_weight_ = (
            None
        )
        x_79 = torch.nn.functional.batch_norm(
            x_78,
            l_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_78 = l_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_80 = torch.conv2d(
            x_79,
            l_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_79 = l_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_81 = torch.nn.functional.batch_norm(
            x_80,
            l_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_80 = l_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_82 = torch.nn.functional.relu(x_81, inplace=True)
        x_81 = None
        x_83 = torch.conv2d(
            x_82,
            l_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            640,
        )
        x_82 = l_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_84 = torch.nn.functional.batch_norm(
            x_83,
            l_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_83 = l_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_85 = torch.nn.functional.relu(x_84, inplace=True)
        x_84 = None
        x_86 = torch.conv2d(
            x_85,
            l_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_85 = l_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_87 = torch.nn.functional.batch_norm(
            x_86,
            l_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_86 = l_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_88 = x_87 + x_77
        x_87 = x_77 = None
        x_89 = torch.conv2d(
            x_88,
            l_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            160,
        )
        l_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_conv_parameters_weight_ = (
            None
        )
        x_90 = torch.nn.functional.batch_norm(
            x_89,
            l_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_89 = l_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_91 = torch.conv2d(
            x_90,
            l_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_90 = l_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_92 = torch.nn.functional.batch_norm(
            x_91,
            l_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_91 = l_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_93 = torch.nn.functional.relu(x_92, inplace=True)
        x_92 = None
        x_94 = torch.conv2d(
            x_93,
            l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_93 = l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_95 = torch.nn.functional.batch_norm(
            x_94,
            l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_94 = l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_96 = x_95 + x_88
        x_95 = x_88 = None
        x_97 = torch.conv2d(
            x_96,
            l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_conv_parameters_weight_ = (
            None
        )
        x_98 = torch.nn.functional.batch_norm(
            x_97,
            l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_97 = l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_99 = torch.nn.functional.relu(x_98, inplace=True)
        x_98 = None
        x_100 = torch.conv2d(
            x_99,
            l_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_99 = l_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_101 = torch.nn.functional.batch_norm(
            x_100,
            l_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_100 = l_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_102 = x_101 + x_96
        x_101 = x_96 = None
        x_103 = torch.conv2d(
            x_102,
            l_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            160,
        )
        l_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_conv_parameters_weight_ = (
            None
        )
        x_104 = torch.nn.functional.batch_norm(
            x_103,
            l_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_103 = l_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_105 = torch.conv2d(
            x_104,
            l_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_104 = l_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_106 = torch.nn.functional.batch_norm(
            x_105,
            l_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_105 = l_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_107 = torch.nn.functional.relu(x_106, inplace=True)
        x_106 = None
        x_108 = torch.conv2d(
            x_107,
            l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_107 = l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_109 = torch.nn.functional.batch_norm(
            x_108,
            l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_108 = l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_110 = x_109 + x_102
        x_109 = x_102 = None
        x_111 = torch.conv2d(
            x_110,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            160,
        )
        x_110 = l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_conv_parameters_weight_ = (None)
        x_112 = torch.nn.functional.batch_norm(
            x_111,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_111 = l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_113 = torch.conv2d(
            x_112,
            l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_112 = l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_114 = torch.nn.functional.batch_norm(
            x_113,
            l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_113 = l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_115 = torch.nn.functional.relu(x_114, inplace=True)
        x_114 = None
        x_116 = torch.conv2d(
            x_115,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            960,
        )
        x_115 = l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_117 = torch.nn.functional.batch_norm(
            x_116,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_116 = l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_118 = torch.nn.functional.relu(x_117, inplace=True)
        x_117 = None
        x_119 = torch._C._nn.pad(x_118, [1, 1, 1, 1], "constant", None)
        x_118 = None
        x_120 = torch.conv2d(
            x_119,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_aa_buffers_filt_,
            stride=2,
            groups=960,
        )
        x_119 = l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_aa_buffers_filt_ = (None)
        x_121 = torch.conv2d(
            x_120,
            l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_120 = l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_122 = torch.nn.functional.batch_norm(
            x_121,
            l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_121 = l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_123 = torch.conv2d(
            x_122,
            l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            256,
        )
        l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_conv_parameters_weight_ = (
            None
        )
        x_124 = torch.nn.functional.batch_norm(
            x_123,
            l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_123 = l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_125 = torch.conv2d(
            x_124,
            l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_124 = l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_126 = torch.nn.functional.batch_norm(
            x_125,
            l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_125 = l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_127 = torch.nn.functional.relu(x_126, inplace=True)
        x_126 = None
        x_128 = torch.conv2d(
            x_127,
            l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1024,
        )
        x_127 = l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_129 = torch.nn.functional.batch_norm(
            x_128,
            l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_128 = l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_130 = torch.nn.functional.relu(x_129, inplace=True)
        x_129 = None
        x_131 = torch.conv2d(
            x_130,
            l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_130 = l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_132 = torch.nn.functional.batch_norm(
            x_131,
            l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_131 = l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_133 = x_132 + x_122
        x_132 = x_122 = None
        x_134 = torch.conv2d(
            x_133,
            l_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        l_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_conv_parameters_weight_ = (
            None
        )
        x_135 = torch.nn.functional.batch_norm(
            x_134,
            l_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_134 = l_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_136 = torch.conv2d(
            x_135,
            l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_135 = l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_137 = torch.nn.functional.batch_norm(
            x_136,
            l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_136 = l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_138 = torch.nn.functional.relu(x_137, inplace=True)
        x_137 = None
        x_139 = torch.conv2d(
            x_138,
            l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1024,
        )
        x_138 = l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_140 = torch.nn.functional.batch_norm(
            x_139,
            l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_139 = l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_141 = torch.nn.functional.relu(x_140, inplace=True)
        x_140 = None
        x_142 = torch.conv2d(
            x_141,
            l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_141 = l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_143 = torch.nn.functional.batch_norm(
            x_142,
            l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_142 = l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_144 = x_143 + x_133
        x_143 = x_133 = None
        x_145 = torch.conv2d(
            x_144,
            l_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        l_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_conv_parameters_weight_ = (
            None
        )
        x_146 = torch.nn.functional.batch_norm(
            x_145,
            l_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_145 = l_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_147 = torch.conv2d(
            x_146,
            l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_146 = l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_148 = torch.nn.functional.batch_norm(
            x_147,
            l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_147 = l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_149 = torch.nn.functional.relu(x_148, inplace=True)
        x_148 = None
        x_150 = torch.conv2d(
            x_149,
            l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1024,
        )
        x_149 = l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_151 = torch.nn.functional.batch_norm(
            x_150,
            l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_150 = l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_152 = torch.nn.functional.relu(x_151, inplace=True)
        x_151 = None
        x_153 = torch.conv2d(
            x_152,
            l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_152 = l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_154 = torch.nn.functional.batch_norm(
            x_153,
            l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_153 = l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_155 = x_154 + x_144
        x_154 = x_144 = None
        x_156 = torch.conv2d(
            x_155,
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
        x_157 = torch.nn.functional.batch_norm(
            x_156,
            l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_156 = l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_158 = torch.nn.functional.relu(x_157, inplace=True)
        x_157 = None
        x_159 = torch.conv2d(
            x_158,
            l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_158 = l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_160 = torch.nn.functional.batch_norm(
            x_159,
            l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_159 = l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_161 = x_160 + x_155
        x_160 = x_155 = None
        x_162 = torch.conv2d(
            x_161,
            l_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        l_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_conv_parameters_weight_ = (
            None
        )
        x_163 = torch.nn.functional.batch_norm(
            x_162,
            l_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_162 = l_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_164 = torch.conv2d(
            x_163,
            l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_163 = l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_165 = torch.nn.functional.batch_norm(
            x_164,
            l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_164 = l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_166 = torch.nn.functional.relu(x_165, inplace=True)
        x_165 = None
        x_167 = torch.conv2d(
            x_166,
            l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_166 = l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_168 = torch.nn.functional.batch_norm(
            x_167,
            l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_167 = l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_169 = x_168 + x_161
        x_168 = x_161 = None
        x_170 = torch.conv2d(
            x_169,
            l_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        l_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_conv_parameters_weight_ = (
            None
        )
        x_171 = torch.nn.functional.batch_norm(
            x_170,
            l_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_170 = l_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_172 = torch.conv2d(
            x_171,
            l_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_171 = l_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_173 = torch.nn.functional.batch_norm(
            x_172,
            l_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_172 = l_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_174 = torch.nn.functional.relu(x_173, inplace=True)
        x_173 = None
        x_175 = torch.conv2d(
            x_174,
            l_self_modules_blocks_modules_3_modules_6_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            512,
        )
        x_174 = l_self_modules_blocks_modules_3_modules_6_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_176 = torch.nn.functional.batch_norm(
            x_175,
            l_self_modules_blocks_modules_3_modules_6_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_6_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_6_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_6_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_175 = l_self_modules_blocks_modules_3_modules_6_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_6_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_6_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_6_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_177 = torch.nn.functional.relu(x_176, inplace=True)
        x_176 = None
        x_178 = torch.conv2d(
            x_177,
            l_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_177 = l_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_179 = torch.nn.functional.batch_norm(
            x_178,
            l_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_178 = l_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_180 = x_179 + x_169
        x_179 = x_169 = None
        x_181 = torch.conv2d(
            x_180,
            l_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            256,
        )
        l_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_conv_parameters_weight_ = (
            None
        )
        x_182 = torch.nn.functional.batch_norm(
            x_181,
            l_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_181 = l_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_183 = torch.conv2d(
            x_182,
            l_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_182 = l_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_184 = torch.nn.functional.batch_norm(
            x_183,
            l_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_183 = l_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_185 = torch.nn.functional.relu(x_184, inplace=True)
        x_184 = None
        x_186 = torch.conv2d(
            x_185,
            l_self_modules_blocks_modules_3_modules_7_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1024,
        )
        x_185 = l_self_modules_blocks_modules_3_modules_7_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_187 = torch.nn.functional.batch_norm(
            x_186,
            l_self_modules_blocks_modules_3_modules_7_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_7_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_7_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_7_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_186 = l_self_modules_blocks_modules_3_modules_7_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_7_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_7_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_7_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_188 = torch.nn.functional.relu(x_187, inplace=True)
        x_187 = None
        x_189 = torch.conv2d(
            x_188,
            l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_188 = l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_190 = torch.nn.functional.batch_norm(
            x_189,
            l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_189 = l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_191 = x_190 + x_180
        x_190 = x_180 = None
        x_192 = torch.conv2d(
            x_191,
            l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_conv_parameters_weight_ = (
            None
        )
        x_193 = torch.nn.functional.batch_norm(
            x_192,
            l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_192 = l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_194 = torch.nn.functional.relu(x_193, inplace=True)
        x_193 = None
        x_195 = torch.conv2d(
            x_194,
            l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_194 = l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_196 = torch.nn.functional.batch_norm(
            x_195,
            l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_195 = l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_197 = x_196 + x_191
        x_196 = x_191 = None
        x_198 = torch.conv2d(
            x_197,
            l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_conv_parameters_weight_ = (
            None
        )
        x_199 = torch.nn.functional.batch_norm(
            x_198,
            l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_198 = l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_200 = torch.nn.functional.relu(x_199, inplace=True)
        x_199 = None
        x_201 = torch.conv2d(
            x_200,
            l_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_200 = l_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_202 = torch.nn.functional.batch_norm(
            x_201,
            l_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_201 = l_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_203 = x_202 + x_197
        x_202 = x_197 = None
        x_204 = torch.conv2d(
            x_203,
            l_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            256,
        )
        l_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_conv_parameters_weight_ = (
            None
        )
        x_205 = torch.nn.functional.batch_norm(
            x_204,
            l_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_204 = l_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_206 = torch.conv2d(
            x_205,
            l_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_205 = l_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_207 = torch.nn.functional.batch_norm(
            x_206,
            l_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_206 = l_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_208 = torch.nn.functional.relu(x_207, inplace=True)
        x_207 = None
        x_209 = torch.conv2d(
            x_208,
            l_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_208 = l_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_210 = torch.nn.functional.batch_norm(
            x_209,
            l_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_209 = l_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_211 = x_210 + x_203
        x_210 = x_203 = None
        x_212 = torch.conv2d(
            x_211,
            l_self_modules_blocks_modules_4_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_211 = (
            l_self_modules_blocks_modules_4_modules_0_modules_conv_parameters_weight_
        ) = None
        x_213 = torch.nn.functional.batch_norm(
            x_212,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_212 = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_214 = torch.nn.functional.relu(x_213, inplace=True)
        x_213 = None
        x_215 = torch.nn.functional.adaptive_avg_pool2d(x_214, 1)
        x_214 = None
        x_216 = torch.conv2d(
            x_215,
            l_self_modules_conv_head_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_215 = l_self_modules_conv_head_parameters_weight_ = None
        x_217 = torch.nn.functional.batch_norm(
            x_216,
            l_self_modules_norm_head_buffers_running_mean_,
            l_self_modules_norm_head_buffers_running_var_,
            l_self_modules_norm_head_parameters_weight_,
            l_self_modules_norm_head_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_216 = (
            l_self_modules_norm_head_buffers_running_mean_
        ) = (
            l_self_modules_norm_head_buffers_running_var_
        ) = (
            l_self_modules_norm_head_parameters_weight_
        ) = l_self_modules_norm_head_parameters_bias_ = None
        x_218 = torch.nn.functional.relu(x_217, inplace=True)
        x_217 = None
        x_219 = x_218.flatten(1, -1)
        x_218 = None
        x_220 = torch._C._nn.linear(
            x_219,
            l_self_modules_classifier_parameters_weight_,
            l_self_modules_classifier_parameters_bias_,
        )
        x_219 = (
            l_self_modules_classifier_parameters_weight_
        ) = l_self_modules_classifier_parameters_bias_ = None
        return (x_220,)
