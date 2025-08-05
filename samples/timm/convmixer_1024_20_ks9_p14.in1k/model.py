import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_stem_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        s1: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_stem_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stem_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_stem_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_0_modules_fn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_0_modules_fn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_0_modules_fn_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_0_modules_fn_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_0_modules_fn_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_0_modules_fn_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_fn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_fn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_fn_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_fn_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_fn_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_fn_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_fn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_fn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_fn_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_fn_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_fn_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_fn_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_fn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_fn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_fn_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_0_modules_fn_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_0_modules_fn_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_fn_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_0_modules_fn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_0_modules_fn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_0_modules_fn_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_0_modules_fn_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_0_modules_fn_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_0_modules_fn_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_fn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_fn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_fn_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_0_modules_fn_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_0_modules_fn_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_fn_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_0_modules_fn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_0_modules_fn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_0_modules_fn_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_0_modules_fn_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_0_modules_fn_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_0_modules_fn_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_0_modules_fn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_0_modules_fn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_0_modules_fn_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_7_modules_0_modules_fn_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_7_modules_0_modules_fn_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_0_modules_fn_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_7_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_7_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_0_modules_fn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_0_modules_fn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_0_modules_fn_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_0_modules_fn_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_0_modules_fn_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_0_modules_fn_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_0_modules_fn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_0_modules_fn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_0_modules_fn_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_9_modules_0_modules_fn_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_9_modules_0_modules_fn_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_0_modules_fn_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_9_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_9_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_0_modules_fn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_0_modules_fn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_0_modules_fn_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_10_modules_0_modules_fn_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_10_modules_0_modules_fn_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_0_modules_fn_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_10_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_10_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_0_modules_fn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_0_modules_fn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_0_modules_fn_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_11_modules_0_modules_fn_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_11_modules_0_modules_fn_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_0_modules_fn_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_11_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_11_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_0_modules_fn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_0_modules_fn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_0_modules_fn_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_12_modules_0_modules_fn_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_12_modules_0_modules_fn_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_0_modules_fn_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_12_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_12_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_0_modules_fn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_0_modules_fn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_0_modules_fn_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_13_modules_0_modules_fn_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_13_modules_0_modules_fn_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_0_modules_fn_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_13_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_13_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_0_modules_fn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_0_modules_fn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_0_modules_fn_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_14_modules_0_modules_fn_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_14_modules_0_modules_fn_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_0_modules_fn_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_14_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_14_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_0_modules_fn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_0_modules_fn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_0_modules_fn_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_15_modules_0_modules_fn_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_15_modules_0_modules_fn_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_0_modules_fn_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_15_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_15_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_0_modules_fn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_0_modules_fn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_0_modules_fn_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_16_modules_0_modules_fn_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_16_modules_0_modules_fn_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_0_modules_fn_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_16_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_16_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_0_modules_fn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_0_modules_fn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_0_modules_fn_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_17_modules_0_modules_fn_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_17_modules_0_modules_fn_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_0_modules_fn_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_17_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_17_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_0_modules_fn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_0_modules_fn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_0_modules_fn_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_18_modules_0_modules_fn_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_18_modules_0_modules_fn_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_0_modules_fn_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_18_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_18_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_0_modules_fn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_0_modules_fn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_0_modules_fn_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_19_modules_0_modules_fn_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_19_modules_0_modules_fn_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_0_modules_fn_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_19_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_19_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_stem_modules_0_parameters_weight_ = (
            L_self_modules_stem_modules_0_parameters_weight_
        )
        l_self_modules_stem_modules_0_parameters_bias_ = (
            L_self_modules_stem_modules_0_parameters_bias_
        )
        l_x_ = L_x_
        l_self_modules_stem_modules_2_buffers_running_mean_ = (
            L_self_modules_stem_modules_2_buffers_running_mean_
        )
        l_self_modules_stem_modules_2_buffers_running_var_ = (
            L_self_modules_stem_modules_2_buffers_running_var_
        )
        l_self_modules_stem_modules_2_parameters_weight_ = (
            L_self_modules_stem_modules_2_parameters_weight_
        )
        l_self_modules_stem_modules_2_parameters_bias_ = (
            L_self_modules_stem_modules_2_parameters_bias_
        )
        l_self_modules_blocks_modules_0_modules_0_modules_fn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_0_modules_0_modules_fn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_0_modules_0_modules_fn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_0_modules_0_modules_fn_modules_0_parameters_bias_
        l_self_modules_blocks_modules_0_modules_0_modules_fn_modules_2_buffers_running_mean_ = L_self_modules_blocks_modules_0_modules_0_modules_fn_modules_2_buffers_running_mean_
        l_self_modules_blocks_modules_0_modules_0_modules_fn_modules_2_buffers_running_var_ = L_self_modules_blocks_modules_0_modules_0_modules_fn_modules_2_buffers_running_var_
        l_self_modules_blocks_modules_0_modules_0_modules_fn_modules_2_parameters_weight_ = L_self_modules_blocks_modules_0_modules_0_modules_fn_modules_2_parameters_weight_
        l_self_modules_blocks_modules_0_modules_0_modules_fn_modules_2_parameters_bias_ = L_self_modules_blocks_modules_0_modules_0_modules_fn_modules_2_parameters_bias_
        l_self_modules_blocks_modules_0_modules_1_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_1_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_1_parameters_bias_ = (
            L_self_modules_blocks_modules_0_modules_1_parameters_bias_
        )
        l_self_modules_blocks_modules_0_modules_3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_0_modules_3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_0_modules_3_buffers_running_var_ = (
            L_self_modules_blocks_modules_0_modules_3_buffers_running_var_
        )
        l_self_modules_blocks_modules_0_modules_3_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_3_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_3_parameters_bias_ = (
            L_self_modules_blocks_modules_0_modules_3_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_0_modules_fn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_1_modules_0_modules_fn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_1_modules_0_modules_fn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_1_modules_0_modules_fn_modules_0_parameters_bias_
        l_self_modules_blocks_modules_1_modules_0_modules_fn_modules_2_buffers_running_mean_ = L_self_modules_blocks_modules_1_modules_0_modules_fn_modules_2_buffers_running_mean_
        l_self_modules_blocks_modules_1_modules_0_modules_fn_modules_2_buffers_running_var_ = L_self_modules_blocks_modules_1_modules_0_modules_fn_modules_2_buffers_running_var_
        l_self_modules_blocks_modules_1_modules_0_modules_fn_modules_2_parameters_weight_ = L_self_modules_blocks_modules_1_modules_0_modules_fn_modules_2_parameters_weight_
        l_self_modules_blocks_modules_1_modules_0_modules_fn_modules_2_parameters_bias_ = L_self_modules_blocks_modules_1_modules_0_modules_fn_modules_2_parameters_bias_
        l_self_modules_blocks_modules_1_modules_1_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_1_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_1_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_1_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_1_modules_3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_1_modules_3_buffers_running_var_ = (
            L_self_modules_blocks_modules_1_modules_3_buffers_running_var_
        )
        l_self_modules_blocks_modules_1_modules_3_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_3_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_3_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_3_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_0_modules_fn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_2_modules_0_modules_fn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_2_modules_0_modules_fn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_2_modules_0_modules_fn_modules_0_parameters_bias_
        l_self_modules_blocks_modules_2_modules_0_modules_fn_modules_2_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_0_modules_fn_modules_2_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_0_modules_fn_modules_2_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_0_modules_fn_modules_2_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_0_modules_fn_modules_2_parameters_weight_ = L_self_modules_blocks_modules_2_modules_0_modules_fn_modules_2_parameters_weight_
        l_self_modules_blocks_modules_2_modules_0_modules_fn_modules_2_parameters_bias_ = L_self_modules_blocks_modules_2_modules_0_modules_fn_modules_2_parameters_bias_
        l_self_modules_blocks_modules_2_modules_1_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_1_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_1_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_1_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_2_modules_3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_2_modules_3_buffers_running_var_ = (
            L_self_modules_blocks_modules_2_modules_3_buffers_running_var_
        )
        l_self_modules_blocks_modules_2_modules_3_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_3_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_3_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_3_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_0_modules_fn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_3_modules_0_modules_fn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_3_modules_0_modules_fn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_3_modules_0_modules_fn_modules_0_parameters_bias_
        l_self_modules_blocks_modules_3_modules_0_modules_fn_modules_2_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_0_modules_fn_modules_2_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_0_modules_fn_modules_2_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_0_modules_fn_modules_2_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_0_modules_fn_modules_2_parameters_weight_ = L_self_modules_blocks_modules_3_modules_0_modules_fn_modules_2_parameters_weight_
        l_self_modules_blocks_modules_3_modules_0_modules_fn_modules_2_parameters_bias_ = L_self_modules_blocks_modules_3_modules_0_modules_fn_modules_2_parameters_bias_
        l_self_modules_blocks_modules_3_modules_1_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_1_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_1_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_1_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_3_modules_3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_3_modules_3_buffers_running_var_ = (
            L_self_modules_blocks_modules_3_modules_3_buffers_running_var_
        )
        l_self_modules_blocks_modules_3_modules_3_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_3_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_3_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_3_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_0_modules_fn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_4_modules_0_modules_fn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_4_modules_0_modules_fn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_4_modules_0_modules_fn_modules_0_parameters_bias_
        l_self_modules_blocks_modules_4_modules_0_modules_fn_modules_2_buffers_running_mean_ = L_self_modules_blocks_modules_4_modules_0_modules_fn_modules_2_buffers_running_mean_
        l_self_modules_blocks_modules_4_modules_0_modules_fn_modules_2_buffers_running_var_ = L_self_modules_blocks_modules_4_modules_0_modules_fn_modules_2_buffers_running_var_
        l_self_modules_blocks_modules_4_modules_0_modules_fn_modules_2_parameters_weight_ = L_self_modules_blocks_modules_4_modules_0_modules_fn_modules_2_parameters_weight_
        l_self_modules_blocks_modules_4_modules_0_modules_fn_modules_2_parameters_bias_ = L_self_modules_blocks_modules_4_modules_0_modules_fn_modules_2_parameters_bias_
        l_self_modules_blocks_modules_4_modules_1_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_1_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_1_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_1_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_3_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_3_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_3_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_3_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_3_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_3_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_0_modules_fn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_5_modules_0_modules_fn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_5_modules_0_modules_fn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_5_modules_0_modules_fn_modules_0_parameters_bias_
        l_self_modules_blocks_modules_5_modules_0_modules_fn_modules_2_buffers_running_mean_ = L_self_modules_blocks_modules_5_modules_0_modules_fn_modules_2_buffers_running_mean_
        l_self_modules_blocks_modules_5_modules_0_modules_fn_modules_2_buffers_running_var_ = L_self_modules_blocks_modules_5_modules_0_modules_fn_modules_2_buffers_running_var_
        l_self_modules_blocks_modules_5_modules_0_modules_fn_modules_2_parameters_weight_ = L_self_modules_blocks_modules_5_modules_0_modules_fn_modules_2_parameters_weight_
        l_self_modules_blocks_modules_5_modules_0_modules_fn_modules_2_parameters_bias_ = L_self_modules_blocks_modules_5_modules_0_modules_fn_modules_2_parameters_bias_
        l_self_modules_blocks_modules_5_modules_1_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_1_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_1_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_1_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_3_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_3_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_3_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_3_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_3_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_3_parameters_bias_
        )
        l_self_modules_blocks_modules_6_modules_0_modules_fn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_6_modules_0_modules_fn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_6_modules_0_modules_fn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_6_modules_0_modules_fn_modules_0_parameters_bias_
        l_self_modules_blocks_modules_6_modules_0_modules_fn_modules_2_buffers_running_mean_ = L_self_modules_blocks_modules_6_modules_0_modules_fn_modules_2_buffers_running_mean_
        l_self_modules_blocks_modules_6_modules_0_modules_fn_modules_2_buffers_running_var_ = L_self_modules_blocks_modules_6_modules_0_modules_fn_modules_2_buffers_running_var_
        l_self_modules_blocks_modules_6_modules_0_modules_fn_modules_2_parameters_weight_ = L_self_modules_blocks_modules_6_modules_0_modules_fn_modules_2_parameters_weight_
        l_self_modules_blocks_modules_6_modules_0_modules_fn_modules_2_parameters_bias_ = L_self_modules_blocks_modules_6_modules_0_modules_fn_modules_2_parameters_bias_
        l_self_modules_blocks_modules_6_modules_1_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_1_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_1_parameters_bias_ = (
            L_self_modules_blocks_modules_6_modules_1_parameters_bias_
        )
        l_self_modules_blocks_modules_6_modules_3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_6_modules_3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_6_modules_3_buffers_running_var_ = (
            L_self_modules_blocks_modules_6_modules_3_buffers_running_var_
        )
        l_self_modules_blocks_modules_6_modules_3_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_3_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_3_parameters_bias_ = (
            L_self_modules_blocks_modules_6_modules_3_parameters_bias_
        )
        l_self_modules_blocks_modules_7_modules_0_modules_fn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_7_modules_0_modules_fn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_7_modules_0_modules_fn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_7_modules_0_modules_fn_modules_0_parameters_bias_
        l_self_modules_blocks_modules_7_modules_0_modules_fn_modules_2_buffers_running_mean_ = L_self_modules_blocks_modules_7_modules_0_modules_fn_modules_2_buffers_running_mean_
        l_self_modules_blocks_modules_7_modules_0_modules_fn_modules_2_buffers_running_var_ = L_self_modules_blocks_modules_7_modules_0_modules_fn_modules_2_buffers_running_var_
        l_self_modules_blocks_modules_7_modules_0_modules_fn_modules_2_parameters_weight_ = L_self_modules_blocks_modules_7_modules_0_modules_fn_modules_2_parameters_weight_
        l_self_modules_blocks_modules_7_modules_0_modules_fn_modules_2_parameters_bias_ = L_self_modules_blocks_modules_7_modules_0_modules_fn_modules_2_parameters_bias_
        l_self_modules_blocks_modules_7_modules_1_parameters_weight_ = (
            L_self_modules_blocks_modules_7_modules_1_parameters_weight_
        )
        l_self_modules_blocks_modules_7_modules_1_parameters_bias_ = (
            L_self_modules_blocks_modules_7_modules_1_parameters_bias_
        )
        l_self_modules_blocks_modules_7_modules_3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_7_modules_3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_7_modules_3_buffers_running_var_ = (
            L_self_modules_blocks_modules_7_modules_3_buffers_running_var_
        )
        l_self_modules_blocks_modules_7_modules_3_parameters_weight_ = (
            L_self_modules_blocks_modules_7_modules_3_parameters_weight_
        )
        l_self_modules_blocks_modules_7_modules_3_parameters_bias_ = (
            L_self_modules_blocks_modules_7_modules_3_parameters_bias_
        )
        l_self_modules_blocks_modules_8_modules_0_modules_fn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_8_modules_0_modules_fn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_8_modules_0_modules_fn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_8_modules_0_modules_fn_modules_0_parameters_bias_
        l_self_modules_blocks_modules_8_modules_0_modules_fn_modules_2_buffers_running_mean_ = L_self_modules_blocks_modules_8_modules_0_modules_fn_modules_2_buffers_running_mean_
        l_self_modules_blocks_modules_8_modules_0_modules_fn_modules_2_buffers_running_var_ = L_self_modules_blocks_modules_8_modules_0_modules_fn_modules_2_buffers_running_var_
        l_self_modules_blocks_modules_8_modules_0_modules_fn_modules_2_parameters_weight_ = L_self_modules_blocks_modules_8_modules_0_modules_fn_modules_2_parameters_weight_
        l_self_modules_blocks_modules_8_modules_0_modules_fn_modules_2_parameters_bias_ = L_self_modules_blocks_modules_8_modules_0_modules_fn_modules_2_parameters_bias_
        l_self_modules_blocks_modules_8_modules_1_parameters_weight_ = (
            L_self_modules_blocks_modules_8_modules_1_parameters_weight_
        )
        l_self_modules_blocks_modules_8_modules_1_parameters_bias_ = (
            L_self_modules_blocks_modules_8_modules_1_parameters_bias_
        )
        l_self_modules_blocks_modules_8_modules_3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_8_modules_3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_8_modules_3_buffers_running_var_ = (
            L_self_modules_blocks_modules_8_modules_3_buffers_running_var_
        )
        l_self_modules_blocks_modules_8_modules_3_parameters_weight_ = (
            L_self_modules_blocks_modules_8_modules_3_parameters_weight_
        )
        l_self_modules_blocks_modules_8_modules_3_parameters_bias_ = (
            L_self_modules_blocks_modules_8_modules_3_parameters_bias_
        )
        l_self_modules_blocks_modules_9_modules_0_modules_fn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_9_modules_0_modules_fn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_9_modules_0_modules_fn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_9_modules_0_modules_fn_modules_0_parameters_bias_
        l_self_modules_blocks_modules_9_modules_0_modules_fn_modules_2_buffers_running_mean_ = L_self_modules_blocks_modules_9_modules_0_modules_fn_modules_2_buffers_running_mean_
        l_self_modules_blocks_modules_9_modules_0_modules_fn_modules_2_buffers_running_var_ = L_self_modules_blocks_modules_9_modules_0_modules_fn_modules_2_buffers_running_var_
        l_self_modules_blocks_modules_9_modules_0_modules_fn_modules_2_parameters_weight_ = L_self_modules_blocks_modules_9_modules_0_modules_fn_modules_2_parameters_weight_
        l_self_modules_blocks_modules_9_modules_0_modules_fn_modules_2_parameters_bias_ = L_self_modules_blocks_modules_9_modules_0_modules_fn_modules_2_parameters_bias_
        l_self_modules_blocks_modules_9_modules_1_parameters_weight_ = (
            L_self_modules_blocks_modules_9_modules_1_parameters_weight_
        )
        l_self_modules_blocks_modules_9_modules_1_parameters_bias_ = (
            L_self_modules_blocks_modules_9_modules_1_parameters_bias_
        )
        l_self_modules_blocks_modules_9_modules_3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_9_modules_3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_9_modules_3_buffers_running_var_ = (
            L_self_modules_blocks_modules_9_modules_3_buffers_running_var_
        )
        l_self_modules_blocks_modules_9_modules_3_parameters_weight_ = (
            L_self_modules_blocks_modules_9_modules_3_parameters_weight_
        )
        l_self_modules_blocks_modules_9_modules_3_parameters_bias_ = (
            L_self_modules_blocks_modules_9_modules_3_parameters_bias_
        )
        l_self_modules_blocks_modules_10_modules_0_modules_fn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_10_modules_0_modules_fn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_10_modules_0_modules_fn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_10_modules_0_modules_fn_modules_0_parameters_bias_
        l_self_modules_blocks_modules_10_modules_0_modules_fn_modules_2_buffers_running_mean_ = L_self_modules_blocks_modules_10_modules_0_modules_fn_modules_2_buffers_running_mean_
        l_self_modules_blocks_modules_10_modules_0_modules_fn_modules_2_buffers_running_var_ = L_self_modules_blocks_modules_10_modules_0_modules_fn_modules_2_buffers_running_var_
        l_self_modules_blocks_modules_10_modules_0_modules_fn_modules_2_parameters_weight_ = L_self_modules_blocks_modules_10_modules_0_modules_fn_modules_2_parameters_weight_
        l_self_modules_blocks_modules_10_modules_0_modules_fn_modules_2_parameters_bias_ = L_self_modules_blocks_modules_10_modules_0_modules_fn_modules_2_parameters_bias_
        l_self_modules_blocks_modules_10_modules_1_parameters_weight_ = (
            L_self_modules_blocks_modules_10_modules_1_parameters_weight_
        )
        l_self_modules_blocks_modules_10_modules_1_parameters_bias_ = (
            L_self_modules_blocks_modules_10_modules_1_parameters_bias_
        )
        l_self_modules_blocks_modules_10_modules_3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_10_modules_3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_10_modules_3_buffers_running_var_ = (
            L_self_modules_blocks_modules_10_modules_3_buffers_running_var_
        )
        l_self_modules_blocks_modules_10_modules_3_parameters_weight_ = (
            L_self_modules_blocks_modules_10_modules_3_parameters_weight_
        )
        l_self_modules_blocks_modules_10_modules_3_parameters_bias_ = (
            L_self_modules_blocks_modules_10_modules_3_parameters_bias_
        )
        l_self_modules_blocks_modules_11_modules_0_modules_fn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_11_modules_0_modules_fn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_11_modules_0_modules_fn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_11_modules_0_modules_fn_modules_0_parameters_bias_
        l_self_modules_blocks_modules_11_modules_0_modules_fn_modules_2_buffers_running_mean_ = L_self_modules_blocks_modules_11_modules_0_modules_fn_modules_2_buffers_running_mean_
        l_self_modules_blocks_modules_11_modules_0_modules_fn_modules_2_buffers_running_var_ = L_self_modules_blocks_modules_11_modules_0_modules_fn_modules_2_buffers_running_var_
        l_self_modules_blocks_modules_11_modules_0_modules_fn_modules_2_parameters_weight_ = L_self_modules_blocks_modules_11_modules_0_modules_fn_modules_2_parameters_weight_
        l_self_modules_blocks_modules_11_modules_0_modules_fn_modules_2_parameters_bias_ = L_self_modules_blocks_modules_11_modules_0_modules_fn_modules_2_parameters_bias_
        l_self_modules_blocks_modules_11_modules_1_parameters_weight_ = (
            L_self_modules_blocks_modules_11_modules_1_parameters_weight_
        )
        l_self_modules_blocks_modules_11_modules_1_parameters_bias_ = (
            L_self_modules_blocks_modules_11_modules_1_parameters_bias_
        )
        l_self_modules_blocks_modules_11_modules_3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_11_modules_3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_11_modules_3_buffers_running_var_ = (
            L_self_modules_blocks_modules_11_modules_3_buffers_running_var_
        )
        l_self_modules_blocks_modules_11_modules_3_parameters_weight_ = (
            L_self_modules_blocks_modules_11_modules_3_parameters_weight_
        )
        l_self_modules_blocks_modules_11_modules_3_parameters_bias_ = (
            L_self_modules_blocks_modules_11_modules_3_parameters_bias_
        )
        l_self_modules_blocks_modules_12_modules_0_modules_fn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_12_modules_0_modules_fn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_12_modules_0_modules_fn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_12_modules_0_modules_fn_modules_0_parameters_bias_
        l_self_modules_blocks_modules_12_modules_0_modules_fn_modules_2_buffers_running_mean_ = L_self_modules_blocks_modules_12_modules_0_modules_fn_modules_2_buffers_running_mean_
        l_self_modules_blocks_modules_12_modules_0_modules_fn_modules_2_buffers_running_var_ = L_self_modules_blocks_modules_12_modules_0_modules_fn_modules_2_buffers_running_var_
        l_self_modules_blocks_modules_12_modules_0_modules_fn_modules_2_parameters_weight_ = L_self_modules_blocks_modules_12_modules_0_modules_fn_modules_2_parameters_weight_
        l_self_modules_blocks_modules_12_modules_0_modules_fn_modules_2_parameters_bias_ = L_self_modules_blocks_modules_12_modules_0_modules_fn_modules_2_parameters_bias_
        l_self_modules_blocks_modules_12_modules_1_parameters_weight_ = (
            L_self_modules_blocks_modules_12_modules_1_parameters_weight_
        )
        l_self_modules_blocks_modules_12_modules_1_parameters_bias_ = (
            L_self_modules_blocks_modules_12_modules_1_parameters_bias_
        )
        l_self_modules_blocks_modules_12_modules_3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_12_modules_3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_12_modules_3_buffers_running_var_ = (
            L_self_modules_blocks_modules_12_modules_3_buffers_running_var_
        )
        l_self_modules_blocks_modules_12_modules_3_parameters_weight_ = (
            L_self_modules_blocks_modules_12_modules_3_parameters_weight_
        )
        l_self_modules_blocks_modules_12_modules_3_parameters_bias_ = (
            L_self_modules_blocks_modules_12_modules_3_parameters_bias_
        )
        l_self_modules_blocks_modules_13_modules_0_modules_fn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_13_modules_0_modules_fn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_13_modules_0_modules_fn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_13_modules_0_modules_fn_modules_0_parameters_bias_
        l_self_modules_blocks_modules_13_modules_0_modules_fn_modules_2_buffers_running_mean_ = L_self_modules_blocks_modules_13_modules_0_modules_fn_modules_2_buffers_running_mean_
        l_self_modules_blocks_modules_13_modules_0_modules_fn_modules_2_buffers_running_var_ = L_self_modules_blocks_modules_13_modules_0_modules_fn_modules_2_buffers_running_var_
        l_self_modules_blocks_modules_13_modules_0_modules_fn_modules_2_parameters_weight_ = L_self_modules_blocks_modules_13_modules_0_modules_fn_modules_2_parameters_weight_
        l_self_modules_blocks_modules_13_modules_0_modules_fn_modules_2_parameters_bias_ = L_self_modules_blocks_modules_13_modules_0_modules_fn_modules_2_parameters_bias_
        l_self_modules_blocks_modules_13_modules_1_parameters_weight_ = (
            L_self_modules_blocks_modules_13_modules_1_parameters_weight_
        )
        l_self_modules_blocks_modules_13_modules_1_parameters_bias_ = (
            L_self_modules_blocks_modules_13_modules_1_parameters_bias_
        )
        l_self_modules_blocks_modules_13_modules_3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_13_modules_3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_13_modules_3_buffers_running_var_ = (
            L_self_modules_blocks_modules_13_modules_3_buffers_running_var_
        )
        l_self_modules_blocks_modules_13_modules_3_parameters_weight_ = (
            L_self_modules_blocks_modules_13_modules_3_parameters_weight_
        )
        l_self_modules_blocks_modules_13_modules_3_parameters_bias_ = (
            L_self_modules_blocks_modules_13_modules_3_parameters_bias_
        )
        l_self_modules_blocks_modules_14_modules_0_modules_fn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_14_modules_0_modules_fn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_14_modules_0_modules_fn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_14_modules_0_modules_fn_modules_0_parameters_bias_
        l_self_modules_blocks_modules_14_modules_0_modules_fn_modules_2_buffers_running_mean_ = L_self_modules_blocks_modules_14_modules_0_modules_fn_modules_2_buffers_running_mean_
        l_self_modules_blocks_modules_14_modules_0_modules_fn_modules_2_buffers_running_var_ = L_self_modules_blocks_modules_14_modules_0_modules_fn_modules_2_buffers_running_var_
        l_self_modules_blocks_modules_14_modules_0_modules_fn_modules_2_parameters_weight_ = L_self_modules_blocks_modules_14_modules_0_modules_fn_modules_2_parameters_weight_
        l_self_modules_blocks_modules_14_modules_0_modules_fn_modules_2_parameters_bias_ = L_self_modules_blocks_modules_14_modules_0_modules_fn_modules_2_parameters_bias_
        l_self_modules_blocks_modules_14_modules_1_parameters_weight_ = (
            L_self_modules_blocks_modules_14_modules_1_parameters_weight_
        )
        l_self_modules_blocks_modules_14_modules_1_parameters_bias_ = (
            L_self_modules_blocks_modules_14_modules_1_parameters_bias_
        )
        l_self_modules_blocks_modules_14_modules_3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_14_modules_3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_14_modules_3_buffers_running_var_ = (
            L_self_modules_blocks_modules_14_modules_3_buffers_running_var_
        )
        l_self_modules_blocks_modules_14_modules_3_parameters_weight_ = (
            L_self_modules_blocks_modules_14_modules_3_parameters_weight_
        )
        l_self_modules_blocks_modules_14_modules_3_parameters_bias_ = (
            L_self_modules_blocks_modules_14_modules_3_parameters_bias_
        )
        l_self_modules_blocks_modules_15_modules_0_modules_fn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_15_modules_0_modules_fn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_15_modules_0_modules_fn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_15_modules_0_modules_fn_modules_0_parameters_bias_
        l_self_modules_blocks_modules_15_modules_0_modules_fn_modules_2_buffers_running_mean_ = L_self_modules_blocks_modules_15_modules_0_modules_fn_modules_2_buffers_running_mean_
        l_self_modules_blocks_modules_15_modules_0_modules_fn_modules_2_buffers_running_var_ = L_self_modules_blocks_modules_15_modules_0_modules_fn_modules_2_buffers_running_var_
        l_self_modules_blocks_modules_15_modules_0_modules_fn_modules_2_parameters_weight_ = L_self_modules_blocks_modules_15_modules_0_modules_fn_modules_2_parameters_weight_
        l_self_modules_blocks_modules_15_modules_0_modules_fn_modules_2_parameters_bias_ = L_self_modules_blocks_modules_15_modules_0_modules_fn_modules_2_parameters_bias_
        l_self_modules_blocks_modules_15_modules_1_parameters_weight_ = (
            L_self_modules_blocks_modules_15_modules_1_parameters_weight_
        )
        l_self_modules_blocks_modules_15_modules_1_parameters_bias_ = (
            L_self_modules_blocks_modules_15_modules_1_parameters_bias_
        )
        l_self_modules_blocks_modules_15_modules_3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_15_modules_3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_15_modules_3_buffers_running_var_ = (
            L_self_modules_blocks_modules_15_modules_3_buffers_running_var_
        )
        l_self_modules_blocks_modules_15_modules_3_parameters_weight_ = (
            L_self_modules_blocks_modules_15_modules_3_parameters_weight_
        )
        l_self_modules_blocks_modules_15_modules_3_parameters_bias_ = (
            L_self_modules_blocks_modules_15_modules_3_parameters_bias_
        )
        l_self_modules_blocks_modules_16_modules_0_modules_fn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_16_modules_0_modules_fn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_16_modules_0_modules_fn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_16_modules_0_modules_fn_modules_0_parameters_bias_
        l_self_modules_blocks_modules_16_modules_0_modules_fn_modules_2_buffers_running_mean_ = L_self_modules_blocks_modules_16_modules_0_modules_fn_modules_2_buffers_running_mean_
        l_self_modules_blocks_modules_16_modules_0_modules_fn_modules_2_buffers_running_var_ = L_self_modules_blocks_modules_16_modules_0_modules_fn_modules_2_buffers_running_var_
        l_self_modules_blocks_modules_16_modules_0_modules_fn_modules_2_parameters_weight_ = L_self_modules_blocks_modules_16_modules_0_modules_fn_modules_2_parameters_weight_
        l_self_modules_blocks_modules_16_modules_0_modules_fn_modules_2_parameters_bias_ = L_self_modules_blocks_modules_16_modules_0_modules_fn_modules_2_parameters_bias_
        l_self_modules_blocks_modules_16_modules_1_parameters_weight_ = (
            L_self_modules_blocks_modules_16_modules_1_parameters_weight_
        )
        l_self_modules_blocks_modules_16_modules_1_parameters_bias_ = (
            L_self_modules_blocks_modules_16_modules_1_parameters_bias_
        )
        l_self_modules_blocks_modules_16_modules_3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_16_modules_3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_16_modules_3_buffers_running_var_ = (
            L_self_modules_blocks_modules_16_modules_3_buffers_running_var_
        )
        l_self_modules_blocks_modules_16_modules_3_parameters_weight_ = (
            L_self_modules_blocks_modules_16_modules_3_parameters_weight_
        )
        l_self_modules_blocks_modules_16_modules_3_parameters_bias_ = (
            L_self_modules_blocks_modules_16_modules_3_parameters_bias_
        )
        l_self_modules_blocks_modules_17_modules_0_modules_fn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_17_modules_0_modules_fn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_17_modules_0_modules_fn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_17_modules_0_modules_fn_modules_0_parameters_bias_
        l_self_modules_blocks_modules_17_modules_0_modules_fn_modules_2_buffers_running_mean_ = L_self_modules_blocks_modules_17_modules_0_modules_fn_modules_2_buffers_running_mean_
        l_self_modules_blocks_modules_17_modules_0_modules_fn_modules_2_buffers_running_var_ = L_self_modules_blocks_modules_17_modules_0_modules_fn_modules_2_buffers_running_var_
        l_self_modules_blocks_modules_17_modules_0_modules_fn_modules_2_parameters_weight_ = L_self_modules_blocks_modules_17_modules_0_modules_fn_modules_2_parameters_weight_
        l_self_modules_blocks_modules_17_modules_0_modules_fn_modules_2_parameters_bias_ = L_self_modules_blocks_modules_17_modules_0_modules_fn_modules_2_parameters_bias_
        l_self_modules_blocks_modules_17_modules_1_parameters_weight_ = (
            L_self_modules_blocks_modules_17_modules_1_parameters_weight_
        )
        l_self_modules_blocks_modules_17_modules_1_parameters_bias_ = (
            L_self_modules_blocks_modules_17_modules_1_parameters_bias_
        )
        l_self_modules_blocks_modules_17_modules_3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_17_modules_3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_17_modules_3_buffers_running_var_ = (
            L_self_modules_blocks_modules_17_modules_3_buffers_running_var_
        )
        l_self_modules_blocks_modules_17_modules_3_parameters_weight_ = (
            L_self_modules_blocks_modules_17_modules_3_parameters_weight_
        )
        l_self_modules_blocks_modules_17_modules_3_parameters_bias_ = (
            L_self_modules_blocks_modules_17_modules_3_parameters_bias_
        )
        l_self_modules_blocks_modules_18_modules_0_modules_fn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_18_modules_0_modules_fn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_18_modules_0_modules_fn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_18_modules_0_modules_fn_modules_0_parameters_bias_
        l_self_modules_blocks_modules_18_modules_0_modules_fn_modules_2_buffers_running_mean_ = L_self_modules_blocks_modules_18_modules_0_modules_fn_modules_2_buffers_running_mean_
        l_self_modules_blocks_modules_18_modules_0_modules_fn_modules_2_buffers_running_var_ = L_self_modules_blocks_modules_18_modules_0_modules_fn_modules_2_buffers_running_var_
        l_self_modules_blocks_modules_18_modules_0_modules_fn_modules_2_parameters_weight_ = L_self_modules_blocks_modules_18_modules_0_modules_fn_modules_2_parameters_weight_
        l_self_modules_blocks_modules_18_modules_0_modules_fn_modules_2_parameters_bias_ = L_self_modules_blocks_modules_18_modules_0_modules_fn_modules_2_parameters_bias_
        l_self_modules_blocks_modules_18_modules_1_parameters_weight_ = (
            L_self_modules_blocks_modules_18_modules_1_parameters_weight_
        )
        l_self_modules_blocks_modules_18_modules_1_parameters_bias_ = (
            L_self_modules_blocks_modules_18_modules_1_parameters_bias_
        )
        l_self_modules_blocks_modules_18_modules_3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_18_modules_3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_18_modules_3_buffers_running_var_ = (
            L_self_modules_blocks_modules_18_modules_3_buffers_running_var_
        )
        l_self_modules_blocks_modules_18_modules_3_parameters_weight_ = (
            L_self_modules_blocks_modules_18_modules_3_parameters_weight_
        )
        l_self_modules_blocks_modules_18_modules_3_parameters_bias_ = (
            L_self_modules_blocks_modules_18_modules_3_parameters_bias_
        )
        l_self_modules_blocks_modules_19_modules_0_modules_fn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_19_modules_0_modules_fn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_19_modules_0_modules_fn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_19_modules_0_modules_fn_modules_0_parameters_bias_
        l_self_modules_blocks_modules_19_modules_0_modules_fn_modules_2_buffers_running_mean_ = L_self_modules_blocks_modules_19_modules_0_modules_fn_modules_2_buffers_running_mean_
        l_self_modules_blocks_modules_19_modules_0_modules_fn_modules_2_buffers_running_var_ = L_self_modules_blocks_modules_19_modules_0_modules_fn_modules_2_buffers_running_var_
        l_self_modules_blocks_modules_19_modules_0_modules_fn_modules_2_parameters_weight_ = L_self_modules_blocks_modules_19_modules_0_modules_fn_modules_2_parameters_weight_
        l_self_modules_blocks_modules_19_modules_0_modules_fn_modules_2_parameters_bias_ = L_self_modules_blocks_modules_19_modules_0_modules_fn_modules_2_parameters_bias_
        l_self_modules_blocks_modules_19_modules_1_parameters_weight_ = (
            L_self_modules_blocks_modules_19_modules_1_parameters_weight_
        )
        l_self_modules_blocks_modules_19_modules_1_parameters_bias_ = (
            L_self_modules_blocks_modules_19_modules_1_parameters_bias_
        )
        l_self_modules_blocks_modules_19_modules_3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_19_modules_3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_19_modules_3_buffers_running_var_ = (
            L_self_modules_blocks_modules_19_modules_3_buffers_running_var_
        )
        l_self_modules_blocks_modules_19_modules_3_parameters_weight_ = (
            L_self_modules_blocks_modules_19_modules_3_parameters_weight_
        )
        l_self_modules_blocks_modules_19_modules_3_parameters_bias_ = (
            L_self_modules_blocks_modules_19_modules_3_parameters_bias_
        )
        l_self_modules_head_parameters_weight_ = L_self_modules_head_parameters_weight_
        l_self_modules_head_parameters_bias_ = L_self_modules_head_parameters_bias_
        input_1 = torch.conv2d(
            l_x_,
            l_self_modules_stem_modules_0_parameters_weight_,
            l_self_modules_stem_modules_0_parameters_bias_,
            (14, 14),
            (0, 0),
            (1, 1),
            1,
        )
        l_x_ = (
            l_self_modules_stem_modules_0_parameters_weight_
        ) = l_self_modules_stem_modules_0_parameters_bias_ = None
        input_2 = torch._C._nn.gelu(input_1, approximate="none")
        input_1 = None
        input_3 = torch.nn.functional.batch_norm(
            input_2,
            l_self_modules_stem_modules_2_buffers_running_mean_,
            l_self_modules_stem_modules_2_buffers_running_var_,
            l_self_modules_stem_modules_2_parameters_weight_,
            l_self_modules_stem_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_2 = (
            l_self_modules_stem_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_stem_modules_2_buffers_running_var_
        ) = (
            l_self_modules_stem_modules_2_parameters_weight_
        ) = l_self_modules_stem_modules_2_parameters_bias_ = None
        input_4 = torch.conv2d(
            input_3,
            l_self_modules_blocks_modules_0_modules_0_modules_fn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_0_modules_fn_modules_0_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            1024,
        )
        l_self_modules_blocks_modules_0_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_0_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_5 = torch._C._nn.gelu(input_4, approximate="none")
        input_4 = None
        input_6 = torch.nn.functional.batch_norm(
            input_5,
            l_self_modules_blocks_modules_0_modules_0_modules_fn_modules_2_buffers_running_mean_,
            l_self_modules_blocks_modules_0_modules_0_modules_fn_modules_2_buffers_running_var_,
            l_self_modules_blocks_modules_0_modules_0_modules_fn_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_0_modules_fn_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_5 = l_self_modules_blocks_modules_0_modules_0_modules_fn_modules_2_buffers_running_mean_ = l_self_modules_blocks_modules_0_modules_0_modules_fn_modules_2_buffers_running_var_ = l_self_modules_blocks_modules_0_modules_0_modules_fn_modules_2_parameters_weight_ = l_self_modules_blocks_modules_0_modules_0_modules_fn_modules_2_parameters_bias_ = (None)
        input_7 = input_6 + input_3
        input_6 = input_3 = None
        input_8 = torch.conv2d(
            input_7,
            l_self_modules_blocks_modules_0_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_7 = (
            l_self_modules_blocks_modules_0_modules_1_parameters_weight_
        ) = l_self_modules_blocks_modules_0_modules_1_parameters_bias_ = None
        input_9 = torch._C._nn.gelu(input_8, approximate="none")
        input_8 = None
        input_10 = torch.nn.functional.batch_norm(
            input_9,
            l_self_modules_blocks_modules_0_modules_3_buffers_running_mean_,
            l_self_modules_blocks_modules_0_modules_3_buffers_running_var_,
            l_self_modules_blocks_modules_0_modules_3_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_9 = (
            l_self_modules_blocks_modules_0_modules_3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_0_modules_3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_0_modules_3_parameters_weight_
        ) = l_self_modules_blocks_modules_0_modules_3_parameters_bias_ = None
        input_11 = torch.conv2d(
            input_10,
            l_self_modules_blocks_modules_1_modules_0_modules_fn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_fn_modules_0_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            1024,
        )
        l_self_modules_blocks_modules_1_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_1_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_12 = torch._C._nn.gelu(input_11, approximate="none")
        input_11 = None
        input_13 = torch.nn.functional.batch_norm(
            input_12,
            l_self_modules_blocks_modules_1_modules_0_modules_fn_modules_2_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_0_modules_fn_modules_2_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_0_modules_fn_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_fn_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_12 = l_self_modules_blocks_modules_1_modules_0_modules_fn_modules_2_buffers_running_mean_ = l_self_modules_blocks_modules_1_modules_0_modules_fn_modules_2_buffers_running_var_ = l_self_modules_blocks_modules_1_modules_0_modules_fn_modules_2_parameters_weight_ = l_self_modules_blocks_modules_1_modules_0_modules_fn_modules_2_parameters_bias_ = (None)
        input_14 = input_13 + input_10
        input_13 = input_10 = None
        input_15 = torch.conv2d(
            input_14,
            l_self_modules_blocks_modules_1_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_14 = (
            l_self_modules_blocks_modules_1_modules_1_parameters_weight_
        ) = l_self_modules_blocks_modules_1_modules_1_parameters_bias_ = None
        input_16 = torch._C._nn.gelu(input_15, approximate="none")
        input_15 = None
        input_17 = torch.nn.functional.batch_norm(
            input_16,
            l_self_modules_blocks_modules_1_modules_3_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_3_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_3_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_16 = (
            l_self_modules_blocks_modules_1_modules_3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_3_parameters_weight_
        ) = l_self_modules_blocks_modules_1_modules_3_parameters_bias_ = None
        input_18 = torch.conv2d(
            input_17,
            l_self_modules_blocks_modules_2_modules_0_modules_fn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_fn_modules_0_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            1024,
        )
        l_self_modules_blocks_modules_2_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_2_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_19 = torch._C._nn.gelu(input_18, approximate="none")
        input_18 = None
        input_20 = torch.nn.functional.batch_norm(
            input_19,
            l_self_modules_blocks_modules_2_modules_0_modules_fn_modules_2_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_0_modules_fn_modules_2_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_0_modules_fn_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_fn_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_19 = l_self_modules_blocks_modules_2_modules_0_modules_fn_modules_2_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_0_modules_fn_modules_2_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_0_modules_fn_modules_2_parameters_weight_ = l_self_modules_blocks_modules_2_modules_0_modules_fn_modules_2_parameters_bias_ = (None)
        input_21 = input_20 + input_17
        input_20 = input_17 = None
        input_22 = torch.conv2d(
            input_21,
            l_self_modules_blocks_modules_2_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_21 = (
            l_self_modules_blocks_modules_2_modules_1_parameters_weight_
        ) = l_self_modules_blocks_modules_2_modules_1_parameters_bias_ = None
        input_23 = torch._C._nn.gelu(input_22, approximate="none")
        input_22 = None
        input_24 = torch.nn.functional.batch_norm(
            input_23,
            l_self_modules_blocks_modules_2_modules_3_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_3_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_3_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_23 = (
            l_self_modules_blocks_modules_2_modules_3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_3_parameters_weight_
        ) = l_self_modules_blocks_modules_2_modules_3_parameters_bias_ = None
        input_25 = torch.conv2d(
            input_24,
            l_self_modules_blocks_modules_3_modules_0_modules_fn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_fn_modules_0_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            1024,
        )
        l_self_modules_blocks_modules_3_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_3_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_26 = torch._C._nn.gelu(input_25, approximate="none")
        input_25 = None
        input_27 = torch.nn.functional.batch_norm(
            input_26,
            l_self_modules_blocks_modules_3_modules_0_modules_fn_modules_2_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_fn_modules_2_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_fn_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_fn_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_26 = l_self_modules_blocks_modules_3_modules_0_modules_fn_modules_2_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_0_modules_fn_modules_2_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_0_modules_fn_modules_2_parameters_weight_ = l_self_modules_blocks_modules_3_modules_0_modules_fn_modules_2_parameters_bias_ = (None)
        input_28 = input_27 + input_24
        input_27 = input_24 = None
        input_29 = torch.conv2d(
            input_28,
            l_self_modules_blocks_modules_3_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_28 = (
            l_self_modules_blocks_modules_3_modules_1_parameters_weight_
        ) = l_self_modules_blocks_modules_3_modules_1_parameters_bias_ = None
        input_30 = torch._C._nn.gelu(input_29, approximate="none")
        input_29 = None
        input_31 = torch.nn.functional.batch_norm(
            input_30,
            l_self_modules_blocks_modules_3_modules_3_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_3_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_3_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_30 = (
            l_self_modules_blocks_modules_3_modules_3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_parameters_weight_
        ) = l_self_modules_blocks_modules_3_modules_3_parameters_bias_ = None
        input_32 = torch.conv2d(
            input_31,
            l_self_modules_blocks_modules_4_modules_0_modules_fn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_0_modules_fn_modules_0_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            1024,
        )
        l_self_modules_blocks_modules_4_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_4_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_33 = torch._C._nn.gelu(input_32, approximate="none")
        input_32 = None
        input_34 = torch.nn.functional.batch_norm(
            input_33,
            l_self_modules_blocks_modules_4_modules_0_modules_fn_modules_2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_0_modules_fn_modules_2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_0_modules_fn_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_0_modules_fn_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_33 = l_self_modules_blocks_modules_4_modules_0_modules_fn_modules_2_buffers_running_mean_ = l_self_modules_blocks_modules_4_modules_0_modules_fn_modules_2_buffers_running_var_ = l_self_modules_blocks_modules_4_modules_0_modules_fn_modules_2_parameters_weight_ = l_self_modules_blocks_modules_4_modules_0_modules_fn_modules_2_parameters_bias_ = (None)
        input_35 = input_34 + input_31
        input_34 = input_31 = None
        input_36 = torch.conv2d(
            input_35,
            l_self_modules_blocks_modules_4_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_35 = (
            l_self_modules_blocks_modules_4_modules_1_parameters_weight_
        ) = l_self_modules_blocks_modules_4_modules_1_parameters_bias_ = None
        input_37 = torch._C._nn.gelu(input_36, approximate="none")
        input_36 = None
        input_38 = torch.nn.functional.batch_norm(
            input_37,
            l_self_modules_blocks_modules_4_modules_3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_37 = (
            l_self_modules_blocks_modules_4_modules_3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_parameters_weight_
        ) = l_self_modules_blocks_modules_4_modules_3_parameters_bias_ = None
        input_39 = torch.conv2d(
            input_38,
            l_self_modules_blocks_modules_5_modules_0_modules_fn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_fn_modules_0_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            1024,
        )
        l_self_modules_blocks_modules_5_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_5_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_40 = torch._C._nn.gelu(input_39, approximate="none")
        input_39 = None
        input_41 = torch.nn.functional.batch_norm(
            input_40,
            l_self_modules_blocks_modules_5_modules_0_modules_fn_modules_2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_0_modules_fn_modules_2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_0_modules_fn_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_fn_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_40 = l_self_modules_blocks_modules_5_modules_0_modules_fn_modules_2_buffers_running_mean_ = l_self_modules_blocks_modules_5_modules_0_modules_fn_modules_2_buffers_running_var_ = l_self_modules_blocks_modules_5_modules_0_modules_fn_modules_2_parameters_weight_ = l_self_modules_blocks_modules_5_modules_0_modules_fn_modules_2_parameters_bias_ = (None)
        input_42 = input_41 + input_38
        input_41 = input_38 = None
        input_43 = torch.conv2d(
            input_42,
            l_self_modules_blocks_modules_5_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_42 = (
            l_self_modules_blocks_modules_5_modules_1_parameters_weight_
        ) = l_self_modules_blocks_modules_5_modules_1_parameters_bias_ = None
        input_44 = torch._C._nn.gelu(input_43, approximate="none")
        input_43 = None
        input_45 = torch.nn.functional.batch_norm(
            input_44,
            l_self_modules_blocks_modules_5_modules_3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_44 = (
            l_self_modules_blocks_modules_5_modules_3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_parameters_weight_
        ) = l_self_modules_blocks_modules_5_modules_3_parameters_bias_ = None
        input_46 = torch.conv2d(
            input_45,
            l_self_modules_blocks_modules_6_modules_0_modules_fn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_0_modules_fn_modules_0_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            1024,
        )
        l_self_modules_blocks_modules_6_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_6_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_47 = torch._C._nn.gelu(input_46, approximate="none")
        input_46 = None
        input_48 = torch.nn.functional.batch_norm(
            input_47,
            l_self_modules_blocks_modules_6_modules_0_modules_fn_modules_2_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_0_modules_fn_modules_2_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_0_modules_fn_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_0_modules_fn_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_47 = l_self_modules_blocks_modules_6_modules_0_modules_fn_modules_2_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_0_modules_fn_modules_2_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_0_modules_fn_modules_2_parameters_weight_ = l_self_modules_blocks_modules_6_modules_0_modules_fn_modules_2_parameters_bias_ = (None)
        input_49 = input_48 + input_45
        input_48 = input_45 = None
        input_50 = torch.conv2d(
            input_49,
            l_self_modules_blocks_modules_6_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_49 = (
            l_self_modules_blocks_modules_6_modules_1_parameters_weight_
        ) = l_self_modules_blocks_modules_6_modules_1_parameters_bias_ = None
        input_51 = torch._C._nn.gelu(input_50, approximate="none")
        input_50 = None
        input_52 = torch.nn.functional.batch_norm(
            input_51,
            l_self_modules_blocks_modules_6_modules_3_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_3_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_3_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_51 = (
            l_self_modules_blocks_modules_6_modules_3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_6_modules_3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_6_modules_3_parameters_weight_
        ) = l_self_modules_blocks_modules_6_modules_3_parameters_bias_ = None
        input_53 = torch.conv2d(
            input_52,
            l_self_modules_blocks_modules_7_modules_0_modules_fn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_0_modules_fn_modules_0_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            1024,
        )
        l_self_modules_blocks_modules_7_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_7_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_54 = torch._C._nn.gelu(input_53, approximate="none")
        input_53 = None
        input_55 = torch.nn.functional.batch_norm(
            input_54,
            l_self_modules_blocks_modules_7_modules_0_modules_fn_modules_2_buffers_running_mean_,
            l_self_modules_blocks_modules_7_modules_0_modules_fn_modules_2_buffers_running_var_,
            l_self_modules_blocks_modules_7_modules_0_modules_fn_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_0_modules_fn_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_54 = l_self_modules_blocks_modules_7_modules_0_modules_fn_modules_2_buffers_running_mean_ = l_self_modules_blocks_modules_7_modules_0_modules_fn_modules_2_buffers_running_var_ = l_self_modules_blocks_modules_7_modules_0_modules_fn_modules_2_parameters_weight_ = l_self_modules_blocks_modules_7_modules_0_modules_fn_modules_2_parameters_bias_ = (None)
        input_56 = input_55 + input_52
        input_55 = input_52 = None
        input_57 = torch.conv2d(
            input_56,
            l_self_modules_blocks_modules_7_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_56 = (
            l_self_modules_blocks_modules_7_modules_1_parameters_weight_
        ) = l_self_modules_blocks_modules_7_modules_1_parameters_bias_ = None
        input_58 = torch._C._nn.gelu(input_57, approximate="none")
        input_57 = None
        input_59 = torch.nn.functional.batch_norm(
            input_58,
            l_self_modules_blocks_modules_7_modules_3_buffers_running_mean_,
            l_self_modules_blocks_modules_7_modules_3_buffers_running_var_,
            l_self_modules_blocks_modules_7_modules_3_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_58 = (
            l_self_modules_blocks_modules_7_modules_3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_7_modules_3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_7_modules_3_parameters_weight_
        ) = l_self_modules_blocks_modules_7_modules_3_parameters_bias_ = None
        input_60 = torch.conv2d(
            input_59,
            l_self_modules_blocks_modules_8_modules_0_modules_fn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_0_modules_fn_modules_0_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            1024,
        )
        l_self_modules_blocks_modules_8_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_8_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_61 = torch._C._nn.gelu(input_60, approximate="none")
        input_60 = None
        input_62 = torch.nn.functional.batch_norm(
            input_61,
            l_self_modules_blocks_modules_8_modules_0_modules_fn_modules_2_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_0_modules_fn_modules_2_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_0_modules_fn_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_0_modules_fn_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_61 = l_self_modules_blocks_modules_8_modules_0_modules_fn_modules_2_buffers_running_mean_ = l_self_modules_blocks_modules_8_modules_0_modules_fn_modules_2_buffers_running_var_ = l_self_modules_blocks_modules_8_modules_0_modules_fn_modules_2_parameters_weight_ = l_self_modules_blocks_modules_8_modules_0_modules_fn_modules_2_parameters_bias_ = (None)
        input_63 = input_62 + input_59
        input_62 = input_59 = None
        input_64 = torch.conv2d(
            input_63,
            l_self_modules_blocks_modules_8_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_63 = (
            l_self_modules_blocks_modules_8_modules_1_parameters_weight_
        ) = l_self_modules_blocks_modules_8_modules_1_parameters_bias_ = None
        input_65 = torch._C._nn.gelu(input_64, approximate="none")
        input_64 = None
        input_66 = torch.nn.functional.batch_norm(
            input_65,
            l_self_modules_blocks_modules_8_modules_3_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_3_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_3_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_65 = (
            l_self_modules_blocks_modules_8_modules_3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_8_modules_3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_8_modules_3_parameters_weight_
        ) = l_self_modules_blocks_modules_8_modules_3_parameters_bias_ = None
        input_67 = torch.conv2d(
            input_66,
            l_self_modules_blocks_modules_9_modules_0_modules_fn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_0_modules_fn_modules_0_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            1024,
        )
        l_self_modules_blocks_modules_9_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_9_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_68 = torch._C._nn.gelu(input_67, approximate="none")
        input_67 = None
        input_69 = torch.nn.functional.batch_norm(
            input_68,
            l_self_modules_blocks_modules_9_modules_0_modules_fn_modules_2_buffers_running_mean_,
            l_self_modules_blocks_modules_9_modules_0_modules_fn_modules_2_buffers_running_var_,
            l_self_modules_blocks_modules_9_modules_0_modules_fn_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_0_modules_fn_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_68 = l_self_modules_blocks_modules_9_modules_0_modules_fn_modules_2_buffers_running_mean_ = l_self_modules_blocks_modules_9_modules_0_modules_fn_modules_2_buffers_running_var_ = l_self_modules_blocks_modules_9_modules_0_modules_fn_modules_2_parameters_weight_ = l_self_modules_blocks_modules_9_modules_0_modules_fn_modules_2_parameters_bias_ = (None)
        input_70 = input_69 + input_66
        input_69 = input_66 = None
        input_71 = torch.conv2d(
            input_70,
            l_self_modules_blocks_modules_9_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_70 = (
            l_self_modules_blocks_modules_9_modules_1_parameters_weight_
        ) = l_self_modules_blocks_modules_9_modules_1_parameters_bias_ = None
        input_72 = torch._C._nn.gelu(input_71, approximate="none")
        input_71 = None
        input_73 = torch.nn.functional.batch_norm(
            input_72,
            l_self_modules_blocks_modules_9_modules_3_buffers_running_mean_,
            l_self_modules_blocks_modules_9_modules_3_buffers_running_var_,
            l_self_modules_blocks_modules_9_modules_3_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_72 = (
            l_self_modules_blocks_modules_9_modules_3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_9_modules_3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_9_modules_3_parameters_weight_
        ) = l_self_modules_blocks_modules_9_modules_3_parameters_bias_ = None
        input_74 = torch.conv2d(
            input_73,
            l_self_modules_blocks_modules_10_modules_0_modules_fn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_0_modules_fn_modules_0_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            1024,
        )
        l_self_modules_blocks_modules_10_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_10_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_75 = torch._C._nn.gelu(input_74, approximate="none")
        input_74 = None
        input_76 = torch.nn.functional.batch_norm(
            input_75,
            l_self_modules_blocks_modules_10_modules_0_modules_fn_modules_2_buffers_running_mean_,
            l_self_modules_blocks_modules_10_modules_0_modules_fn_modules_2_buffers_running_var_,
            l_self_modules_blocks_modules_10_modules_0_modules_fn_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_0_modules_fn_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_75 = l_self_modules_blocks_modules_10_modules_0_modules_fn_modules_2_buffers_running_mean_ = l_self_modules_blocks_modules_10_modules_0_modules_fn_modules_2_buffers_running_var_ = l_self_modules_blocks_modules_10_modules_0_modules_fn_modules_2_parameters_weight_ = l_self_modules_blocks_modules_10_modules_0_modules_fn_modules_2_parameters_bias_ = (None)
        input_77 = input_76 + input_73
        input_76 = input_73 = None
        input_78 = torch.conv2d(
            input_77,
            l_self_modules_blocks_modules_10_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_77 = (
            l_self_modules_blocks_modules_10_modules_1_parameters_weight_
        ) = l_self_modules_blocks_modules_10_modules_1_parameters_bias_ = None
        input_79 = torch._C._nn.gelu(input_78, approximate="none")
        input_78 = None
        input_80 = torch.nn.functional.batch_norm(
            input_79,
            l_self_modules_blocks_modules_10_modules_3_buffers_running_mean_,
            l_self_modules_blocks_modules_10_modules_3_buffers_running_var_,
            l_self_modules_blocks_modules_10_modules_3_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_79 = (
            l_self_modules_blocks_modules_10_modules_3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_10_modules_3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_10_modules_3_parameters_weight_
        ) = l_self_modules_blocks_modules_10_modules_3_parameters_bias_ = None
        input_81 = torch.conv2d(
            input_80,
            l_self_modules_blocks_modules_11_modules_0_modules_fn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_0_modules_fn_modules_0_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            1024,
        )
        l_self_modules_blocks_modules_11_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_11_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_82 = torch._C._nn.gelu(input_81, approximate="none")
        input_81 = None
        input_83 = torch.nn.functional.batch_norm(
            input_82,
            l_self_modules_blocks_modules_11_modules_0_modules_fn_modules_2_buffers_running_mean_,
            l_self_modules_blocks_modules_11_modules_0_modules_fn_modules_2_buffers_running_var_,
            l_self_modules_blocks_modules_11_modules_0_modules_fn_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_0_modules_fn_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_82 = l_self_modules_blocks_modules_11_modules_0_modules_fn_modules_2_buffers_running_mean_ = l_self_modules_blocks_modules_11_modules_0_modules_fn_modules_2_buffers_running_var_ = l_self_modules_blocks_modules_11_modules_0_modules_fn_modules_2_parameters_weight_ = l_self_modules_blocks_modules_11_modules_0_modules_fn_modules_2_parameters_bias_ = (None)
        input_84 = input_83 + input_80
        input_83 = input_80 = None
        input_85 = torch.conv2d(
            input_84,
            l_self_modules_blocks_modules_11_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_84 = (
            l_self_modules_blocks_modules_11_modules_1_parameters_weight_
        ) = l_self_modules_blocks_modules_11_modules_1_parameters_bias_ = None
        input_86 = torch._C._nn.gelu(input_85, approximate="none")
        input_85 = None
        input_87 = torch.nn.functional.batch_norm(
            input_86,
            l_self_modules_blocks_modules_11_modules_3_buffers_running_mean_,
            l_self_modules_blocks_modules_11_modules_3_buffers_running_var_,
            l_self_modules_blocks_modules_11_modules_3_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_86 = (
            l_self_modules_blocks_modules_11_modules_3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_11_modules_3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_11_modules_3_parameters_weight_
        ) = l_self_modules_blocks_modules_11_modules_3_parameters_bias_ = None
        input_88 = torch.conv2d(
            input_87,
            l_self_modules_blocks_modules_12_modules_0_modules_fn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_0_modules_fn_modules_0_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            1024,
        )
        l_self_modules_blocks_modules_12_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_12_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_89 = torch._C._nn.gelu(input_88, approximate="none")
        input_88 = None
        input_90 = torch.nn.functional.batch_norm(
            input_89,
            l_self_modules_blocks_modules_12_modules_0_modules_fn_modules_2_buffers_running_mean_,
            l_self_modules_blocks_modules_12_modules_0_modules_fn_modules_2_buffers_running_var_,
            l_self_modules_blocks_modules_12_modules_0_modules_fn_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_0_modules_fn_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_89 = l_self_modules_blocks_modules_12_modules_0_modules_fn_modules_2_buffers_running_mean_ = l_self_modules_blocks_modules_12_modules_0_modules_fn_modules_2_buffers_running_var_ = l_self_modules_blocks_modules_12_modules_0_modules_fn_modules_2_parameters_weight_ = l_self_modules_blocks_modules_12_modules_0_modules_fn_modules_2_parameters_bias_ = (None)
        input_91 = input_90 + input_87
        input_90 = input_87 = None
        input_92 = torch.conv2d(
            input_91,
            l_self_modules_blocks_modules_12_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_91 = (
            l_self_modules_blocks_modules_12_modules_1_parameters_weight_
        ) = l_self_modules_blocks_modules_12_modules_1_parameters_bias_ = None
        input_93 = torch._C._nn.gelu(input_92, approximate="none")
        input_92 = None
        input_94 = torch.nn.functional.batch_norm(
            input_93,
            l_self_modules_blocks_modules_12_modules_3_buffers_running_mean_,
            l_self_modules_blocks_modules_12_modules_3_buffers_running_var_,
            l_self_modules_blocks_modules_12_modules_3_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_93 = (
            l_self_modules_blocks_modules_12_modules_3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_12_modules_3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_12_modules_3_parameters_weight_
        ) = l_self_modules_blocks_modules_12_modules_3_parameters_bias_ = None
        input_95 = torch.conv2d(
            input_94,
            l_self_modules_blocks_modules_13_modules_0_modules_fn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_0_modules_fn_modules_0_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            1024,
        )
        l_self_modules_blocks_modules_13_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_13_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_96 = torch._C._nn.gelu(input_95, approximate="none")
        input_95 = None
        input_97 = torch.nn.functional.batch_norm(
            input_96,
            l_self_modules_blocks_modules_13_modules_0_modules_fn_modules_2_buffers_running_mean_,
            l_self_modules_blocks_modules_13_modules_0_modules_fn_modules_2_buffers_running_var_,
            l_self_modules_blocks_modules_13_modules_0_modules_fn_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_0_modules_fn_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_96 = l_self_modules_blocks_modules_13_modules_0_modules_fn_modules_2_buffers_running_mean_ = l_self_modules_blocks_modules_13_modules_0_modules_fn_modules_2_buffers_running_var_ = l_self_modules_blocks_modules_13_modules_0_modules_fn_modules_2_parameters_weight_ = l_self_modules_blocks_modules_13_modules_0_modules_fn_modules_2_parameters_bias_ = (None)
        input_98 = input_97 + input_94
        input_97 = input_94 = None
        input_99 = torch.conv2d(
            input_98,
            l_self_modules_blocks_modules_13_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_98 = (
            l_self_modules_blocks_modules_13_modules_1_parameters_weight_
        ) = l_self_modules_blocks_modules_13_modules_1_parameters_bias_ = None
        input_100 = torch._C._nn.gelu(input_99, approximate="none")
        input_99 = None
        input_101 = torch.nn.functional.batch_norm(
            input_100,
            l_self_modules_blocks_modules_13_modules_3_buffers_running_mean_,
            l_self_modules_blocks_modules_13_modules_3_buffers_running_var_,
            l_self_modules_blocks_modules_13_modules_3_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_100 = (
            l_self_modules_blocks_modules_13_modules_3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_13_modules_3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_13_modules_3_parameters_weight_
        ) = l_self_modules_blocks_modules_13_modules_3_parameters_bias_ = None
        input_102 = torch.conv2d(
            input_101,
            l_self_modules_blocks_modules_14_modules_0_modules_fn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_0_modules_fn_modules_0_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            1024,
        )
        l_self_modules_blocks_modules_14_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_14_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_103 = torch._C._nn.gelu(input_102, approximate="none")
        input_102 = None
        input_104 = torch.nn.functional.batch_norm(
            input_103,
            l_self_modules_blocks_modules_14_modules_0_modules_fn_modules_2_buffers_running_mean_,
            l_self_modules_blocks_modules_14_modules_0_modules_fn_modules_2_buffers_running_var_,
            l_self_modules_blocks_modules_14_modules_0_modules_fn_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_0_modules_fn_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_103 = l_self_modules_blocks_modules_14_modules_0_modules_fn_modules_2_buffers_running_mean_ = l_self_modules_blocks_modules_14_modules_0_modules_fn_modules_2_buffers_running_var_ = l_self_modules_blocks_modules_14_modules_0_modules_fn_modules_2_parameters_weight_ = l_self_modules_blocks_modules_14_modules_0_modules_fn_modules_2_parameters_bias_ = (None)
        input_105 = input_104 + input_101
        input_104 = input_101 = None
        input_106 = torch.conv2d(
            input_105,
            l_self_modules_blocks_modules_14_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_105 = (
            l_self_modules_blocks_modules_14_modules_1_parameters_weight_
        ) = l_self_modules_blocks_modules_14_modules_1_parameters_bias_ = None
        input_107 = torch._C._nn.gelu(input_106, approximate="none")
        input_106 = None
        input_108 = torch.nn.functional.batch_norm(
            input_107,
            l_self_modules_blocks_modules_14_modules_3_buffers_running_mean_,
            l_self_modules_blocks_modules_14_modules_3_buffers_running_var_,
            l_self_modules_blocks_modules_14_modules_3_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_107 = (
            l_self_modules_blocks_modules_14_modules_3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_14_modules_3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_14_modules_3_parameters_weight_
        ) = l_self_modules_blocks_modules_14_modules_3_parameters_bias_ = None
        input_109 = torch.conv2d(
            input_108,
            l_self_modules_blocks_modules_15_modules_0_modules_fn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_0_modules_fn_modules_0_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            1024,
        )
        l_self_modules_blocks_modules_15_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_15_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_110 = torch._C._nn.gelu(input_109, approximate="none")
        input_109 = None
        input_111 = torch.nn.functional.batch_norm(
            input_110,
            l_self_modules_blocks_modules_15_modules_0_modules_fn_modules_2_buffers_running_mean_,
            l_self_modules_blocks_modules_15_modules_0_modules_fn_modules_2_buffers_running_var_,
            l_self_modules_blocks_modules_15_modules_0_modules_fn_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_0_modules_fn_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_110 = l_self_modules_blocks_modules_15_modules_0_modules_fn_modules_2_buffers_running_mean_ = l_self_modules_blocks_modules_15_modules_0_modules_fn_modules_2_buffers_running_var_ = l_self_modules_blocks_modules_15_modules_0_modules_fn_modules_2_parameters_weight_ = l_self_modules_blocks_modules_15_modules_0_modules_fn_modules_2_parameters_bias_ = (None)
        input_112 = input_111 + input_108
        input_111 = input_108 = None
        input_113 = torch.conv2d(
            input_112,
            l_self_modules_blocks_modules_15_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_112 = (
            l_self_modules_blocks_modules_15_modules_1_parameters_weight_
        ) = l_self_modules_blocks_modules_15_modules_1_parameters_bias_ = None
        input_114 = torch._C._nn.gelu(input_113, approximate="none")
        input_113 = None
        input_115 = torch.nn.functional.batch_norm(
            input_114,
            l_self_modules_blocks_modules_15_modules_3_buffers_running_mean_,
            l_self_modules_blocks_modules_15_modules_3_buffers_running_var_,
            l_self_modules_blocks_modules_15_modules_3_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_114 = (
            l_self_modules_blocks_modules_15_modules_3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_15_modules_3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_15_modules_3_parameters_weight_
        ) = l_self_modules_blocks_modules_15_modules_3_parameters_bias_ = None
        input_116 = torch.conv2d(
            input_115,
            l_self_modules_blocks_modules_16_modules_0_modules_fn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_0_modules_fn_modules_0_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            1024,
        )
        l_self_modules_blocks_modules_16_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_16_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_117 = torch._C._nn.gelu(input_116, approximate="none")
        input_116 = None
        input_118 = torch.nn.functional.batch_norm(
            input_117,
            l_self_modules_blocks_modules_16_modules_0_modules_fn_modules_2_buffers_running_mean_,
            l_self_modules_blocks_modules_16_modules_0_modules_fn_modules_2_buffers_running_var_,
            l_self_modules_blocks_modules_16_modules_0_modules_fn_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_0_modules_fn_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_117 = l_self_modules_blocks_modules_16_modules_0_modules_fn_modules_2_buffers_running_mean_ = l_self_modules_blocks_modules_16_modules_0_modules_fn_modules_2_buffers_running_var_ = l_self_modules_blocks_modules_16_modules_0_modules_fn_modules_2_parameters_weight_ = l_self_modules_blocks_modules_16_modules_0_modules_fn_modules_2_parameters_bias_ = (None)
        input_119 = input_118 + input_115
        input_118 = input_115 = None
        input_120 = torch.conv2d(
            input_119,
            l_self_modules_blocks_modules_16_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_119 = (
            l_self_modules_blocks_modules_16_modules_1_parameters_weight_
        ) = l_self_modules_blocks_modules_16_modules_1_parameters_bias_ = None
        input_121 = torch._C._nn.gelu(input_120, approximate="none")
        input_120 = None
        input_122 = torch.nn.functional.batch_norm(
            input_121,
            l_self_modules_blocks_modules_16_modules_3_buffers_running_mean_,
            l_self_modules_blocks_modules_16_modules_3_buffers_running_var_,
            l_self_modules_blocks_modules_16_modules_3_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_121 = (
            l_self_modules_blocks_modules_16_modules_3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_16_modules_3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_16_modules_3_parameters_weight_
        ) = l_self_modules_blocks_modules_16_modules_3_parameters_bias_ = None
        input_123 = torch.conv2d(
            input_122,
            l_self_modules_blocks_modules_17_modules_0_modules_fn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_0_modules_fn_modules_0_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            1024,
        )
        l_self_modules_blocks_modules_17_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_17_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_124 = torch._C._nn.gelu(input_123, approximate="none")
        input_123 = None
        input_125 = torch.nn.functional.batch_norm(
            input_124,
            l_self_modules_blocks_modules_17_modules_0_modules_fn_modules_2_buffers_running_mean_,
            l_self_modules_blocks_modules_17_modules_0_modules_fn_modules_2_buffers_running_var_,
            l_self_modules_blocks_modules_17_modules_0_modules_fn_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_0_modules_fn_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_124 = l_self_modules_blocks_modules_17_modules_0_modules_fn_modules_2_buffers_running_mean_ = l_self_modules_blocks_modules_17_modules_0_modules_fn_modules_2_buffers_running_var_ = l_self_modules_blocks_modules_17_modules_0_modules_fn_modules_2_parameters_weight_ = l_self_modules_blocks_modules_17_modules_0_modules_fn_modules_2_parameters_bias_ = (None)
        input_126 = input_125 + input_122
        input_125 = input_122 = None
        input_127 = torch.conv2d(
            input_126,
            l_self_modules_blocks_modules_17_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_126 = (
            l_self_modules_blocks_modules_17_modules_1_parameters_weight_
        ) = l_self_modules_blocks_modules_17_modules_1_parameters_bias_ = None
        input_128 = torch._C._nn.gelu(input_127, approximate="none")
        input_127 = None
        input_129 = torch.nn.functional.batch_norm(
            input_128,
            l_self_modules_blocks_modules_17_modules_3_buffers_running_mean_,
            l_self_modules_blocks_modules_17_modules_3_buffers_running_var_,
            l_self_modules_blocks_modules_17_modules_3_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_128 = (
            l_self_modules_blocks_modules_17_modules_3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_17_modules_3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_17_modules_3_parameters_weight_
        ) = l_self_modules_blocks_modules_17_modules_3_parameters_bias_ = None
        input_130 = torch.conv2d(
            input_129,
            l_self_modules_blocks_modules_18_modules_0_modules_fn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_0_modules_fn_modules_0_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            1024,
        )
        l_self_modules_blocks_modules_18_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_18_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_131 = torch._C._nn.gelu(input_130, approximate="none")
        input_130 = None
        input_132 = torch.nn.functional.batch_norm(
            input_131,
            l_self_modules_blocks_modules_18_modules_0_modules_fn_modules_2_buffers_running_mean_,
            l_self_modules_blocks_modules_18_modules_0_modules_fn_modules_2_buffers_running_var_,
            l_self_modules_blocks_modules_18_modules_0_modules_fn_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_0_modules_fn_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_131 = l_self_modules_blocks_modules_18_modules_0_modules_fn_modules_2_buffers_running_mean_ = l_self_modules_blocks_modules_18_modules_0_modules_fn_modules_2_buffers_running_var_ = l_self_modules_blocks_modules_18_modules_0_modules_fn_modules_2_parameters_weight_ = l_self_modules_blocks_modules_18_modules_0_modules_fn_modules_2_parameters_bias_ = (None)
        input_133 = input_132 + input_129
        input_132 = input_129 = None
        input_134 = torch.conv2d(
            input_133,
            l_self_modules_blocks_modules_18_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_133 = (
            l_self_modules_blocks_modules_18_modules_1_parameters_weight_
        ) = l_self_modules_blocks_modules_18_modules_1_parameters_bias_ = None
        input_135 = torch._C._nn.gelu(input_134, approximate="none")
        input_134 = None
        input_136 = torch.nn.functional.batch_norm(
            input_135,
            l_self_modules_blocks_modules_18_modules_3_buffers_running_mean_,
            l_self_modules_blocks_modules_18_modules_3_buffers_running_var_,
            l_self_modules_blocks_modules_18_modules_3_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_135 = (
            l_self_modules_blocks_modules_18_modules_3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_18_modules_3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_18_modules_3_parameters_weight_
        ) = l_self_modules_blocks_modules_18_modules_3_parameters_bias_ = None
        input_137 = torch.conv2d(
            input_136,
            l_self_modules_blocks_modules_19_modules_0_modules_fn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_0_modules_fn_modules_0_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            1024,
        )
        l_self_modules_blocks_modules_19_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_19_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_138 = torch._C._nn.gelu(input_137, approximate="none")
        input_137 = None
        input_139 = torch.nn.functional.batch_norm(
            input_138,
            l_self_modules_blocks_modules_19_modules_0_modules_fn_modules_2_buffers_running_mean_,
            l_self_modules_blocks_modules_19_modules_0_modules_fn_modules_2_buffers_running_var_,
            l_self_modules_blocks_modules_19_modules_0_modules_fn_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_0_modules_fn_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_138 = l_self_modules_blocks_modules_19_modules_0_modules_fn_modules_2_buffers_running_mean_ = l_self_modules_blocks_modules_19_modules_0_modules_fn_modules_2_buffers_running_var_ = l_self_modules_blocks_modules_19_modules_0_modules_fn_modules_2_parameters_weight_ = l_self_modules_blocks_modules_19_modules_0_modules_fn_modules_2_parameters_bias_ = (None)
        input_140 = input_139 + input_136
        input_139 = input_136 = None
        input_141 = torch.conv2d(
            input_140,
            l_self_modules_blocks_modules_19_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_140 = (
            l_self_modules_blocks_modules_19_modules_1_parameters_weight_
        ) = l_self_modules_blocks_modules_19_modules_1_parameters_bias_ = None
        input_142 = torch._C._nn.gelu(input_141, approximate="none")
        input_141 = None
        input_143 = torch.nn.functional.batch_norm(
            input_142,
            l_self_modules_blocks_modules_19_modules_3_buffers_running_mean_,
            l_self_modules_blocks_modules_19_modules_3_buffers_running_var_,
            l_self_modules_blocks_modules_19_modules_3_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_142 = (
            l_self_modules_blocks_modules_19_modules_3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_19_modules_3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_19_modules_3_parameters_weight_
        ) = l_self_modules_blocks_modules_19_modules_3_parameters_bias_ = None
        x = torch.nn.functional.adaptive_avg_pool2d(input_143, 1)
        input_143 = None
        x_1 = x.flatten(1, -1)
        x = None
        x_2 = torch.nn.functional.dropout(x_1, 0.0, False, False)
        x_1 = None
        x_3 = torch._C._nn.linear(
            x_2,
            l_self_modules_head_parameters_weight_,
            l_self_modules_head_parameters_bias_,
        )
        x_2 = (
            l_self_modules_head_parameters_weight_
        ) = l_self_modules_head_parameters_bias_ = None
        return (x_3,)
