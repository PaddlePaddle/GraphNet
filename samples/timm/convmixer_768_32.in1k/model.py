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
        L_self_modules_blocks_modules_20_modules_0_modules_fn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_0_modules_fn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_0_modules_fn_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_20_modules_0_modules_fn_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_20_modules_0_modules_fn_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_0_modules_fn_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_20_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_20_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_0_modules_fn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_0_modules_fn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_0_modules_fn_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_21_modules_0_modules_fn_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_21_modules_0_modules_fn_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_0_modules_fn_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_21_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_21_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_0_modules_fn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_0_modules_fn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_0_modules_fn_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_22_modules_0_modules_fn_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_22_modules_0_modules_fn_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_0_modules_fn_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_22_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_22_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_0_modules_fn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_0_modules_fn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_0_modules_fn_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_23_modules_0_modules_fn_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_23_modules_0_modules_fn_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_0_modules_fn_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_23_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_23_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_24_modules_0_modules_fn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_24_modules_0_modules_fn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_24_modules_0_modules_fn_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_24_modules_0_modules_fn_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_24_modules_0_modules_fn_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_24_modules_0_modules_fn_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_24_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_24_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_24_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_24_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_24_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_24_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_25_modules_0_modules_fn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_25_modules_0_modules_fn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_25_modules_0_modules_fn_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_25_modules_0_modules_fn_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_25_modules_0_modules_fn_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_25_modules_0_modules_fn_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_25_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_25_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_25_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_25_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_25_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_25_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_26_modules_0_modules_fn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_26_modules_0_modules_fn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_26_modules_0_modules_fn_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_26_modules_0_modules_fn_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_26_modules_0_modules_fn_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_26_modules_0_modules_fn_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_26_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_26_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_26_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_26_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_26_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_26_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_27_modules_0_modules_fn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_27_modules_0_modules_fn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_27_modules_0_modules_fn_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_27_modules_0_modules_fn_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_27_modules_0_modules_fn_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_27_modules_0_modules_fn_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_27_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_27_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_27_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_27_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_27_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_27_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_28_modules_0_modules_fn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_28_modules_0_modules_fn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_28_modules_0_modules_fn_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_28_modules_0_modules_fn_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_28_modules_0_modules_fn_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_28_modules_0_modules_fn_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_28_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_28_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_28_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_28_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_28_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_28_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_29_modules_0_modules_fn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_29_modules_0_modules_fn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_29_modules_0_modules_fn_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_29_modules_0_modules_fn_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_29_modules_0_modules_fn_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_29_modules_0_modules_fn_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_29_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_29_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_29_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_29_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_29_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_29_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_30_modules_0_modules_fn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_30_modules_0_modules_fn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_30_modules_0_modules_fn_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_30_modules_0_modules_fn_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_30_modules_0_modules_fn_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_30_modules_0_modules_fn_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_30_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_30_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_30_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_30_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_30_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_30_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_31_modules_0_modules_fn_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_31_modules_0_modules_fn_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_31_modules_0_modules_fn_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_31_modules_0_modules_fn_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_31_modules_0_modules_fn_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_31_modules_0_modules_fn_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_31_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_31_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_31_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_31_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_31_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_31_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_blocks_modules_20_modules_0_modules_fn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_20_modules_0_modules_fn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_20_modules_0_modules_fn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_20_modules_0_modules_fn_modules_0_parameters_bias_
        l_self_modules_blocks_modules_20_modules_0_modules_fn_modules_2_buffers_running_mean_ = L_self_modules_blocks_modules_20_modules_0_modules_fn_modules_2_buffers_running_mean_
        l_self_modules_blocks_modules_20_modules_0_modules_fn_modules_2_buffers_running_var_ = L_self_modules_blocks_modules_20_modules_0_modules_fn_modules_2_buffers_running_var_
        l_self_modules_blocks_modules_20_modules_0_modules_fn_modules_2_parameters_weight_ = L_self_modules_blocks_modules_20_modules_0_modules_fn_modules_2_parameters_weight_
        l_self_modules_blocks_modules_20_modules_0_modules_fn_modules_2_parameters_bias_ = L_self_modules_blocks_modules_20_modules_0_modules_fn_modules_2_parameters_bias_
        l_self_modules_blocks_modules_20_modules_1_parameters_weight_ = (
            L_self_modules_blocks_modules_20_modules_1_parameters_weight_
        )
        l_self_modules_blocks_modules_20_modules_1_parameters_bias_ = (
            L_self_modules_blocks_modules_20_modules_1_parameters_bias_
        )
        l_self_modules_blocks_modules_20_modules_3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_20_modules_3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_20_modules_3_buffers_running_var_ = (
            L_self_modules_blocks_modules_20_modules_3_buffers_running_var_
        )
        l_self_modules_blocks_modules_20_modules_3_parameters_weight_ = (
            L_self_modules_blocks_modules_20_modules_3_parameters_weight_
        )
        l_self_modules_blocks_modules_20_modules_3_parameters_bias_ = (
            L_self_modules_blocks_modules_20_modules_3_parameters_bias_
        )
        l_self_modules_blocks_modules_21_modules_0_modules_fn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_21_modules_0_modules_fn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_21_modules_0_modules_fn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_21_modules_0_modules_fn_modules_0_parameters_bias_
        l_self_modules_blocks_modules_21_modules_0_modules_fn_modules_2_buffers_running_mean_ = L_self_modules_blocks_modules_21_modules_0_modules_fn_modules_2_buffers_running_mean_
        l_self_modules_blocks_modules_21_modules_0_modules_fn_modules_2_buffers_running_var_ = L_self_modules_blocks_modules_21_modules_0_modules_fn_modules_2_buffers_running_var_
        l_self_modules_blocks_modules_21_modules_0_modules_fn_modules_2_parameters_weight_ = L_self_modules_blocks_modules_21_modules_0_modules_fn_modules_2_parameters_weight_
        l_self_modules_blocks_modules_21_modules_0_modules_fn_modules_2_parameters_bias_ = L_self_modules_blocks_modules_21_modules_0_modules_fn_modules_2_parameters_bias_
        l_self_modules_blocks_modules_21_modules_1_parameters_weight_ = (
            L_self_modules_blocks_modules_21_modules_1_parameters_weight_
        )
        l_self_modules_blocks_modules_21_modules_1_parameters_bias_ = (
            L_self_modules_blocks_modules_21_modules_1_parameters_bias_
        )
        l_self_modules_blocks_modules_21_modules_3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_21_modules_3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_21_modules_3_buffers_running_var_ = (
            L_self_modules_blocks_modules_21_modules_3_buffers_running_var_
        )
        l_self_modules_blocks_modules_21_modules_3_parameters_weight_ = (
            L_self_modules_blocks_modules_21_modules_3_parameters_weight_
        )
        l_self_modules_blocks_modules_21_modules_3_parameters_bias_ = (
            L_self_modules_blocks_modules_21_modules_3_parameters_bias_
        )
        l_self_modules_blocks_modules_22_modules_0_modules_fn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_22_modules_0_modules_fn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_22_modules_0_modules_fn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_22_modules_0_modules_fn_modules_0_parameters_bias_
        l_self_modules_blocks_modules_22_modules_0_modules_fn_modules_2_buffers_running_mean_ = L_self_modules_blocks_modules_22_modules_0_modules_fn_modules_2_buffers_running_mean_
        l_self_modules_blocks_modules_22_modules_0_modules_fn_modules_2_buffers_running_var_ = L_self_modules_blocks_modules_22_modules_0_modules_fn_modules_2_buffers_running_var_
        l_self_modules_blocks_modules_22_modules_0_modules_fn_modules_2_parameters_weight_ = L_self_modules_blocks_modules_22_modules_0_modules_fn_modules_2_parameters_weight_
        l_self_modules_blocks_modules_22_modules_0_modules_fn_modules_2_parameters_bias_ = L_self_modules_blocks_modules_22_modules_0_modules_fn_modules_2_parameters_bias_
        l_self_modules_blocks_modules_22_modules_1_parameters_weight_ = (
            L_self_modules_blocks_modules_22_modules_1_parameters_weight_
        )
        l_self_modules_blocks_modules_22_modules_1_parameters_bias_ = (
            L_self_modules_blocks_modules_22_modules_1_parameters_bias_
        )
        l_self_modules_blocks_modules_22_modules_3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_22_modules_3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_22_modules_3_buffers_running_var_ = (
            L_self_modules_blocks_modules_22_modules_3_buffers_running_var_
        )
        l_self_modules_blocks_modules_22_modules_3_parameters_weight_ = (
            L_self_modules_blocks_modules_22_modules_3_parameters_weight_
        )
        l_self_modules_blocks_modules_22_modules_3_parameters_bias_ = (
            L_self_modules_blocks_modules_22_modules_3_parameters_bias_
        )
        l_self_modules_blocks_modules_23_modules_0_modules_fn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_23_modules_0_modules_fn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_23_modules_0_modules_fn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_23_modules_0_modules_fn_modules_0_parameters_bias_
        l_self_modules_blocks_modules_23_modules_0_modules_fn_modules_2_buffers_running_mean_ = L_self_modules_blocks_modules_23_modules_0_modules_fn_modules_2_buffers_running_mean_
        l_self_modules_blocks_modules_23_modules_0_modules_fn_modules_2_buffers_running_var_ = L_self_modules_blocks_modules_23_modules_0_modules_fn_modules_2_buffers_running_var_
        l_self_modules_blocks_modules_23_modules_0_modules_fn_modules_2_parameters_weight_ = L_self_modules_blocks_modules_23_modules_0_modules_fn_modules_2_parameters_weight_
        l_self_modules_blocks_modules_23_modules_0_modules_fn_modules_2_parameters_bias_ = L_self_modules_blocks_modules_23_modules_0_modules_fn_modules_2_parameters_bias_
        l_self_modules_blocks_modules_23_modules_1_parameters_weight_ = (
            L_self_modules_blocks_modules_23_modules_1_parameters_weight_
        )
        l_self_modules_blocks_modules_23_modules_1_parameters_bias_ = (
            L_self_modules_blocks_modules_23_modules_1_parameters_bias_
        )
        l_self_modules_blocks_modules_23_modules_3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_23_modules_3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_23_modules_3_buffers_running_var_ = (
            L_self_modules_blocks_modules_23_modules_3_buffers_running_var_
        )
        l_self_modules_blocks_modules_23_modules_3_parameters_weight_ = (
            L_self_modules_blocks_modules_23_modules_3_parameters_weight_
        )
        l_self_modules_blocks_modules_23_modules_3_parameters_bias_ = (
            L_self_modules_blocks_modules_23_modules_3_parameters_bias_
        )
        l_self_modules_blocks_modules_24_modules_0_modules_fn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_24_modules_0_modules_fn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_24_modules_0_modules_fn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_24_modules_0_modules_fn_modules_0_parameters_bias_
        l_self_modules_blocks_modules_24_modules_0_modules_fn_modules_2_buffers_running_mean_ = L_self_modules_blocks_modules_24_modules_0_modules_fn_modules_2_buffers_running_mean_
        l_self_modules_blocks_modules_24_modules_0_modules_fn_modules_2_buffers_running_var_ = L_self_modules_blocks_modules_24_modules_0_modules_fn_modules_2_buffers_running_var_
        l_self_modules_blocks_modules_24_modules_0_modules_fn_modules_2_parameters_weight_ = L_self_modules_blocks_modules_24_modules_0_modules_fn_modules_2_parameters_weight_
        l_self_modules_blocks_modules_24_modules_0_modules_fn_modules_2_parameters_bias_ = L_self_modules_blocks_modules_24_modules_0_modules_fn_modules_2_parameters_bias_
        l_self_modules_blocks_modules_24_modules_1_parameters_weight_ = (
            L_self_modules_blocks_modules_24_modules_1_parameters_weight_
        )
        l_self_modules_blocks_modules_24_modules_1_parameters_bias_ = (
            L_self_modules_blocks_modules_24_modules_1_parameters_bias_
        )
        l_self_modules_blocks_modules_24_modules_3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_24_modules_3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_24_modules_3_buffers_running_var_ = (
            L_self_modules_blocks_modules_24_modules_3_buffers_running_var_
        )
        l_self_modules_blocks_modules_24_modules_3_parameters_weight_ = (
            L_self_modules_blocks_modules_24_modules_3_parameters_weight_
        )
        l_self_modules_blocks_modules_24_modules_3_parameters_bias_ = (
            L_self_modules_blocks_modules_24_modules_3_parameters_bias_
        )
        l_self_modules_blocks_modules_25_modules_0_modules_fn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_25_modules_0_modules_fn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_25_modules_0_modules_fn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_25_modules_0_modules_fn_modules_0_parameters_bias_
        l_self_modules_blocks_modules_25_modules_0_modules_fn_modules_2_buffers_running_mean_ = L_self_modules_blocks_modules_25_modules_0_modules_fn_modules_2_buffers_running_mean_
        l_self_modules_blocks_modules_25_modules_0_modules_fn_modules_2_buffers_running_var_ = L_self_modules_blocks_modules_25_modules_0_modules_fn_modules_2_buffers_running_var_
        l_self_modules_blocks_modules_25_modules_0_modules_fn_modules_2_parameters_weight_ = L_self_modules_blocks_modules_25_modules_0_modules_fn_modules_2_parameters_weight_
        l_self_modules_blocks_modules_25_modules_0_modules_fn_modules_2_parameters_bias_ = L_self_modules_blocks_modules_25_modules_0_modules_fn_modules_2_parameters_bias_
        l_self_modules_blocks_modules_25_modules_1_parameters_weight_ = (
            L_self_modules_blocks_modules_25_modules_1_parameters_weight_
        )
        l_self_modules_blocks_modules_25_modules_1_parameters_bias_ = (
            L_self_modules_blocks_modules_25_modules_1_parameters_bias_
        )
        l_self_modules_blocks_modules_25_modules_3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_25_modules_3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_25_modules_3_buffers_running_var_ = (
            L_self_modules_blocks_modules_25_modules_3_buffers_running_var_
        )
        l_self_modules_blocks_modules_25_modules_3_parameters_weight_ = (
            L_self_modules_blocks_modules_25_modules_3_parameters_weight_
        )
        l_self_modules_blocks_modules_25_modules_3_parameters_bias_ = (
            L_self_modules_blocks_modules_25_modules_3_parameters_bias_
        )
        l_self_modules_blocks_modules_26_modules_0_modules_fn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_26_modules_0_modules_fn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_26_modules_0_modules_fn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_26_modules_0_modules_fn_modules_0_parameters_bias_
        l_self_modules_blocks_modules_26_modules_0_modules_fn_modules_2_buffers_running_mean_ = L_self_modules_blocks_modules_26_modules_0_modules_fn_modules_2_buffers_running_mean_
        l_self_modules_blocks_modules_26_modules_0_modules_fn_modules_2_buffers_running_var_ = L_self_modules_blocks_modules_26_modules_0_modules_fn_modules_2_buffers_running_var_
        l_self_modules_blocks_modules_26_modules_0_modules_fn_modules_2_parameters_weight_ = L_self_modules_blocks_modules_26_modules_0_modules_fn_modules_2_parameters_weight_
        l_self_modules_blocks_modules_26_modules_0_modules_fn_modules_2_parameters_bias_ = L_self_modules_blocks_modules_26_modules_0_modules_fn_modules_2_parameters_bias_
        l_self_modules_blocks_modules_26_modules_1_parameters_weight_ = (
            L_self_modules_blocks_modules_26_modules_1_parameters_weight_
        )
        l_self_modules_blocks_modules_26_modules_1_parameters_bias_ = (
            L_self_modules_blocks_modules_26_modules_1_parameters_bias_
        )
        l_self_modules_blocks_modules_26_modules_3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_26_modules_3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_26_modules_3_buffers_running_var_ = (
            L_self_modules_blocks_modules_26_modules_3_buffers_running_var_
        )
        l_self_modules_blocks_modules_26_modules_3_parameters_weight_ = (
            L_self_modules_blocks_modules_26_modules_3_parameters_weight_
        )
        l_self_modules_blocks_modules_26_modules_3_parameters_bias_ = (
            L_self_modules_blocks_modules_26_modules_3_parameters_bias_
        )
        l_self_modules_blocks_modules_27_modules_0_modules_fn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_27_modules_0_modules_fn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_27_modules_0_modules_fn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_27_modules_0_modules_fn_modules_0_parameters_bias_
        l_self_modules_blocks_modules_27_modules_0_modules_fn_modules_2_buffers_running_mean_ = L_self_modules_blocks_modules_27_modules_0_modules_fn_modules_2_buffers_running_mean_
        l_self_modules_blocks_modules_27_modules_0_modules_fn_modules_2_buffers_running_var_ = L_self_modules_blocks_modules_27_modules_0_modules_fn_modules_2_buffers_running_var_
        l_self_modules_blocks_modules_27_modules_0_modules_fn_modules_2_parameters_weight_ = L_self_modules_blocks_modules_27_modules_0_modules_fn_modules_2_parameters_weight_
        l_self_modules_blocks_modules_27_modules_0_modules_fn_modules_2_parameters_bias_ = L_self_modules_blocks_modules_27_modules_0_modules_fn_modules_2_parameters_bias_
        l_self_modules_blocks_modules_27_modules_1_parameters_weight_ = (
            L_self_modules_blocks_modules_27_modules_1_parameters_weight_
        )
        l_self_modules_blocks_modules_27_modules_1_parameters_bias_ = (
            L_self_modules_blocks_modules_27_modules_1_parameters_bias_
        )
        l_self_modules_blocks_modules_27_modules_3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_27_modules_3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_27_modules_3_buffers_running_var_ = (
            L_self_modules_blocks_modules_27_modules_3_buffers_running_var_
        )
        l_self_modules_blocks_modules_27_modules_3_parameters_weight_ = (
            L_self_modules_blocks_modules_27_modules_3_parameters_weight_
        )
        l_self_modules_blocks_modules_27_modules_3_parameters_bias_ = (
            L_self_modules_blocks_modules_27_modules_3_parameters_bias_
        )
        l_self_modules_blocks_modules_28_modules_0_modules_fn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_28_modules_0_modules_fn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_28_modules_0_modules_fn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_28_modules_0_modules_fn_modules_0_parameters_bias_
        l_self_modules_blocks_modules_28_modules_0_modules_fn_modules_2_buffers_running_mean_ = L_self_modules_blocks_modules_28_modules_0_modules_fn_modules_2_buffers_running_mean_
        l_self_modules_blocks_modules_28_modules_0_modules_fn_modules_2_buffers_running_var_ = L_self_modules_blocks_modules_28_modules_0_modules_fn_modules_2_buffers_running_var_
        l_self_modules_blocks_modules_28_modules_0_modules_fn_modules_2_parameters_weight_ = L_self_modules_blocks_modules_28_modules_0_modules_fn_modules_2_parameters_weight_
        l_self_modules_blocks_modules_28_modules_0_modules_fn_modules_2_parameters_bias_ = L_self_modules_blocks_modules_28_modules_0_modules_fn_modules_2_parameters_bias_
        l_self_modules_blocks_modules_28_modules_1_parameters_weight_ = (
            L_self_modules_blocks_modules_28_modules_1_parameters_weight_
        )
        l_self_modules_blocks_modules_28_modules_1_parameters_bias_ = (
            L_self_modules_blocks_modules_28_modules_1_parameters_bias_
        )
        l_self_modules_blocks_modules_28_modules_3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_28_modules_3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_28_modules_3_buffers_running_var_ = (
            L_self_modules_blocks_modules_28_modules_3_buffers_running_var_
        )
        l_self_modules_blocks_modules_28_modules_3_parameters_weight_ = (
            L_self_modules_blocks_modules_28_modules_3_parameters_weight_
        )
        l_self_modules_blocks_modules_28_modules_3_parameters_bias_ = (
            L_self_modules_blocks_modules_28_modules_3_parameters_bias_
        )
        l_self_modules_blocks_modules_29_modules_0_modules_fn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_29_modules_0_modules_fn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_29_modules_0_modules_fn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_29_modules_0_modules_fn_modules_0_parameters_bias_
        l_self_modules_blocks_modules_29_modules_0_modules_fn_modules_2_buffers_running_mean_ = L_self_modules_blocks_modules_29_modules_0_modules_fn_modules_2_buffers_running_mean_
        l_self_modules_blocks_modules_29_modules_0_modules_fn_modules_2_buffers_running_var_ = L_self_modules_blocks_modules_29_modules_0_modules_fn_modules_2_buffers_running_var_
        l_self_modules_blocks_modules_29_modules_0_modules_fn_modules_2_parameters_weight_ = L_self_modules_blocks_modules_29_modules_0_modules_fn_modules_2_parameters_weight_
        l_self_modules_blocks_modules_29_modules_0_modules_fn_modules_2_parameters_bias_ = L_self_modules_blocks_modules_29_modules_0_modules_fn_modules_2_parameters_bias_
        l_self_modules_blocks_modules_29_modules_1_parameters_weight_ = (
            L_self_modules_blocks_modules_29_modules_1_parameters_weight_
        )
        l_self_modules_blocks_modules_29_modules_1_parameters_bias_ = (
            L_self_modules_blocks_modules_29_modules_1_parameters_bias_
        )
        l_self_modules_blocks_modules_29_modules_3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_29_modules_3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_29_modules_3_buffers_running_var_ = (
            L_self_modules_blocks_modules_29_modules_3_buffers_running_var_
        )
        l_self_modules_blocks_modules_29_modules_3_parameters_weight_ = (
            L_self_modules_blocks_modules_29_modules_3_parameters_weight_
        )
        l_self_modules_blocks_modules_29_modules_3_parameters_bias_ = (
            L_self_modules_blocks_modules_29_modules_3_parameters_bias_
        )
        l_self_modules_blocks_modules_30_modules_0_modules_fn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_30_modules_0_modules_fn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_30_modules_0_modules_fn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_30_modules_0_modules_fn_modules_0_parameters_bias_
        l_self_modules_blocks_modules_30_modules_0_modules_fn_modules_2_buffers_running_mean_ = L_self_modules_blocks_modules_30_modules_0_modules_fn_modules_2_buffers_running_mean_
        l_self_modules_blocks_modules_30_modules_0_modules_fn_modules_2_buffers_running_var_ = L_self_modules_blocks_modules_30_modules_0_modules_fn_modules_2_buffers_running_var_
        l_self_modules_blocks_modules_30_modules_0_modules_fn_modules_2_parameters_weight_ = L_self_modules_blocks_modules_30_modules_0_modules_fn_modules_2_parameters_weight_
        l_self_modules_blocks_modules_30_modules_0_modules_fn_modules_2_parameters_bias_ = L_self_modules_blocks_modules_30_modules_0_modules_fn_modules_2_parameters_bias_
        l_self_modules_blocks_modules_30_modules_1_parameters_weight_ = (
            L_self_modules_blocks_modules_30_modules_1_parameters_weight_
        )
        l_self_modules_blocks_modules_30_modules_1_parameters_bias_ = (
            L_self_modules_blocks_modules_30_modules_1_parameters_bias_
        )
        l_self_modules_blocks_modules_30_modules_3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_30_modules_3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_30_modules_3_buffers_running_var_ = (
            L_self_modules_blocks_modules_30_modules_3_buffers_running_var_
        )
        l_self_modules_blocks_modules_30_modules_3_parameters_weight_ = (
            L_self_modules_blocks_modules_30_modules_3_parameters_weight_
        )
        l_self_modules_blocks_modules_30_modules_3_parameters_bias_ = (
            L_self_modules_blocks_modules_30_modules_3_parameters_bias_
        )
        l_self_modules_blocks_modules_31_modules_0_modules_fn_modules_0_parameters_weight_ = L_self_modules_blocks_modules_31_modules_0_modules_fn_modules_0_parameters_weight_
        l_self_modules_blocks_modules_31_modules_0_modules_fn_modules_0_parameters_bias_ = L_self_modules_blocks_modules_31_modules_0_modules_fn_modules_0_parameters_bias_
        l_self_modules_blocks_modules_31_modules_0_modules_fn_modules_2_buffers_running_mean_ = L_self_modules_blocks_modules_31_modules_0_modules_fn_modules_2_buffers_running_mean_
        l_self_modules_blocks_modules_31_modules_0_modules_fn_modules_2_buffers_running_var_ = L_self_modules_blocks_modules_31_modules_0_modules_fn_modules_2_buffers_running_var_
        l_self_modules_blocks_modules_31_modules_0_modules_fn_modules_2_parameters_weight_ = L_self_modules_blocks_modules_31_modules_0_modules_fn_modules_2_parameters_weight_
        l_self_modules_blocks_modules_31_modules_0_modules_fn_modules_2_parameters_bias_ = L_self_modules_blocks_modules_31_modules_0_modules_fn_modules_2_parameters_bias_
        l_self_modules_blocks_modules_31_modules_1_parameters_weight_ = (
            L_self_modules_blocks_modules_31_modules_1_parameters_weight_
        )
        l_self_modules_blocks_modules_31_modules_1_parameters_bias_ = (
            L_self_modules_blocks_modules_31_modules_1_parameters_bias_
        )
        l_self_modules_blocks_modules_31_modules_3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_31_modules_3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_31_modules_3_buffers_running_var_ = (
            L_self_modules_blocks_modules_31_modules_3_buffers_running_var_
        )
        l_self_modules_blocks_modules_31_modules_3_parameters_weight_ = (
            L_self_modules_blocks_modules_31_modules_3_parameters_weight_
        )
        l_self_modules_blocks_modules_31_modules_3_parameters_bias_ = (
            L_self_modules_blocks_modules_31_modules_3_parameters_bias_
        )
        l_self_modules_head_parameters_weight_ = L_self_modules_head_parameters_weight_
        l_self_modules_head_parameters_bias_ = L_self_modules_head_parameters_bias_
        input_1 = torch.conv2d(
            l_x_,
            l_self_modules_stem_modules_0_parameters_weight_,
            l_self_modules_stem_modules_0_parameters_bias_,
            (7, 7),
            (0, 0),
            (1, 1),
            1,
        )
        l_x_ = (
            l_self_modules_stem_modules_0_parameters_weight_
        ) = l_self_modules_stem_modules_0_parameters_bias_ = None
        input_2 = torch.nn.functional.relu(input_1, inplace=False)
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
            768,
        )
        l_self_modules_blocks_modules_0_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_0_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_5 = torch.nn.functional.relu(input_4, inplace=False)
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
        input_9 = torch.nn.functional.relu(input_8, inplace=False)
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
            768,
        )
        l_self_modules_blocks_modules_1_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_1_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_12 = torch.nn.functional.relu(input_11, inplace=False)
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
        input_16 = torch.nn.functional.relu(input_15, inplace=False)
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
            768,
        )
        l_self_modules_blocks_modules_2_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_2_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_19 = torch.nn.functional.relu(input_18, inplace=False)
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
        input_23 = torch.nn.functional.relu(input_22, inplace=False)
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
            768,
        )
        l_self_modules_blocks_modules_3_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_3_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_26 = torch.nn.functional.relu(input_25, inplace=False)
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
        input_30 = torch.nn.functional.relu(input_29, inplace=False)
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
            768,
        )
        l_self_modules_blocks_modules_4_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_4_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_33 = torch.nn.functional.relu(input_32, inplace=False)
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
        input_37 = torch.nn.functional.relu(input_36, inplace=False)
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
            768,
        )
        l_self_modules_blocks_modules_5_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_5_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_40 = torch.nn.functional.relu(input_39, inplace=False)
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
        input_44 = torch.nn.functional.relu(input_43, inplace=False)
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
            768,
        )
        l_self_modules_blocks_modules_6_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_6_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_47 = torch.nn.functional.relu(input_46, inplace=False)
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
        input_51 = torch.nn.functional.relu(input_50, inplace=False)
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
            768,
        )
        l_self_modules_blocks_modules_7_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_7_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_54 = torch.nn.functional.relu(input_53, inplace=False)
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
        input_58 = torch.nn.functional.relu(input_57, inplace=False)
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
            768,
        )
        l_self_modules_blocks_modules_8_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_8_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_61 = torch.nn.functional.relu(input_60, inplace=False)
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
        input_65 = torch.nn.functional.relu(input_64, inplace=False)
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
            768,
        )
        l_self_modules_blocks_modules_9_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_9_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_68 = torch.nn.functional.relu(input_67, inplace=False)
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
        input_72 = torch.nn.functional.relu(input_71, inplace=False)
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
            768,
        )
        l_self_modules_blocks_modules_10_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_10_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_75 = torch.nn.functional.relu(input_74, inplace=False)
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
        input_79 = torch.nn.functional.relu(input_78, inplace=False)
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
            768,
        )
        l_self_modules_blocks_modules_11_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_11_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_82 = torch.nn.functional.relu(input_81, inplace=False)
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
        input_86 = torch.nn.functional.relu(input_85, inplace=False)
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
            768,
        )
        l_self_modules_blocks_modules_12_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_12_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_89 = torch.nn.functional.relu(input_88, inplace=False)
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
        input_93 = torch.nn.functional.relu(input_92, inplace=False)
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
            768,
        )
        l_self_modules_blocks_modules_13_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_13_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_96 = torch.nn.functional.relu(input_95, inplace=False)
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
        input_100 = torch.nn.functional.relu(input_99, inplace=False)
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
            768,
        )
        l_self_modules_blocks_modules_14_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_14_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_103 = torch.nn.functional.relu(input_102, inplace=False)
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
        input_107 = torch.nn.functional.relu(input_106, inplace=False)
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
            768,
        )
        l_self_modules_blocks_modules_15_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_15_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_110 = torch.nn.functional.relu(input_109, inplace=False)
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
        input_114 = torch.nn.functional.relu(input_113, inplace=False)
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
            768,
        )
        l_self_modules_blocks_modules_16_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_16_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_117 = torch.nn.functional.relu(input_116, inplace=False)
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
        input_121 = torch.nn.functional.relu(input_120, inplace=False)
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
            768,
        )
        l_self_modules_blocks_modules_17_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_17_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_124 = torch.nn.functional.relu(input_123, inplace=False)
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
        input_128 = torch.nn.functional.relu(input_127, inplace=False)
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
            768,
        )
        l_self_modules_blocks_modules_18_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_18_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_131 = torch.nn.functional.relu(input_130, inplace=False)
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
        input_135 = torch.nn.functional.relu(input_134, inplace=False)
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
            768,
        )
        l_self_modules_blocks_modules_19_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_19_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_138 = torch.nn.functional.relu(input_137, inplace=False)
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
        input_142 = torch.nn.functional.relu(input_141, inplace=False)
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
        input_144 = torch.conv2d(
            input_143,
            l_self_modules_blocks_modules_20_modules_0_modules_fn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_0_modules_fn_modules_0_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            768,
        )
        l_self_modules_blocks_modules_20_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_20_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_145 = torch.nn.functional.relu(input_144, inplace=False)
        input_144 = None
        input_146 = torch.nn.functional.batch_norm(
            input_145,
            l_self_modules_blocks_modules_20_modules_0_modules_fn_modules_2_buffers_running_mean_,
            l_self_modules_blocks_modules_20_modules_0_modules_fn_modules_2_buffers_running_var_,
            l_self_modules_blocks_modules_20_modules_0_modules_fn_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_0_modules_fn_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_145 = l_self_modules_blocks_modules_20_modules_0_modules_fn_modules_2_buffers_running_mean_ = l_self_modules_blocks_modules_20_modules_0_modules_fn_modules_2_buffers_running_var_ = l_self_modules_blocks_modules_20_modules_0_modules_fn_modules_2_parameters_weight_ = l_self_modules_blocks_modules_20_modules_0_modules_fn_modules_2_parameters_bias_ = (None)
        input_147 = input_146 + input_143
        input_146 = input_143 = None
        input_148 = torch.conv2d(
            input_147,
            l_self_modules_blocks_modules_20_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_147 = (
            l_self_modules_blocks_modules_20_modules_1_parameters_weight_
        ) = l_self_modules_blocks_modules_20_modules_1_parameters_bias_ = None
        input_149 = torch.nn.functional.relu(input_148, inplace=False)
        input_148 = None
        input_150 = torch.nn.functional.batch_norm(
            input_149,
            l_self_modules_blocks_modules_20_modules_3_buffers_running_mean_,
            l_self_modules_blocks_modules_20_modules_3_buffers_running_var_,
            l_self_modules_blocks_modules_20_modules_3_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_149 = (
            l_self_modules_blocks_modules_20_modules_3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_20_modules_3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_20_modules_3_parameters_weight_
        ) = l_self_modules_blocks_modules_20_modules_3_parameters_bias_ = None
        input_151 = torch.conv2d(
            input_150,
            l_self_modules_blocks_modules_21_modules_0_modules_fn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_0_modules_fn_modules_0_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            768,
        )
        l_self_modules_blocks_modules_21_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_21_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_152 = torch.nn.functional.relu(input_151, inplace=False)
        input_151 = None
        input_153 = torch.nn.functional.batch_norm(
            input_152,
            l_self_modules_blocks_modules_21_modules_0_modules_fn_modules_2_buffers_running_mean_,
            l_self_modules_blocks_modules_21_modules_0_modules_fn_modules_2_buffers_running_var_,
            l_self_modules_blocks_modules_21_modules_0_modules_fn_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_0_modules_fn_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_152 = l_self_modules_blocks_modules_21_modules_0_modules_fn_modules_2_buffers_running_mean_ = l_self_modules_blocks_modules_21_modules_0_modules_fn_modules_2_buffers_running_var_ = l_self_modules_blocks_modules_21_modules_0_modules_fn_modules_2_parameters_weight_ = l_self_modules_blocks_modules_21_modules_0_modules_fn_modules_2_parameters_bias_ = (None)
        input_154 = input_153 + input_150
        input_153 = input_150 = None
        input_155 = torch.conv2d(
            input_154,
            l_self_modules_blocks_modules_21_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_154 = (
            l_self_modules_blocks_modules_21_modules_1_parameters_weight_
        ) = l_self_modules_blocks_modules_21_modules_1_parameters_bias_ = None
        input_156 = torch.nn.functional.relu(input_155, inplace=False)
        input_155 = None
        input_157 = torch.nn.functional.batch_norm(
            input_156,
            l_self_modules_blocks_modules_21_modules_3_buffers_running_mean_,
            l_self_modules_blocks_modules_21_modules_3_buffers_running_var_,
            l_self_modules_blocks_modules_21_modules_3_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_156 = (
            l_self_modules_blocks_modules_21_modules_3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_21_modules_3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_21_modules_3_parameters_weight_
        ) = l_self_modules_blocks_modules_21_modules_3_parameters_bias_ = None
        input_158 = torch.conv2d(
            input_157,
            l_self_modules_blocks_modules_22_modules_0_modules_fn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_0_modules_fn_modules_0_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            768,
        )
        l_self_modules_blocks_modules_22_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_22_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_159 = torch.nn.functional.relu(input_158, inplace=False)
        input_158 = None
        input_160 = torch.nn.functional.batch_norm(
            input_159,
            l_self_modules_blocks_modules_22_modules_0_modules_fn_modules_2_buffers_running_mean_,
            l_self_modules_blocks_modules_22_modules_0_modules_fn_modules_2_buffers_running_var_,
            l_self_modules_blocks_modules_22_modules_0_modules_fn_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_0_modules_fn_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_159 = l_self_modules_blocks_modules_22_modules_0_modules_fn_modules_2_buffers_running_mean_ = l_self_modules_blocks_modules_22_modules_0_modules_fn_modules_2_buffers_running_var_ = l_self_modules_blocks_modules_22_modules_0_modules_fn_modules_2_parameters_weight_ = l_self_modules_blocks_modules_22_modules_0_modules_fn_modules_2_parameters_bias_ = (None)
        input_161 = input_160 + input_157
        input_160 = input_157 = None
        input_162 = torch.conv2d(
            input_161,
            l_self_modules_blocks_modules_22_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_161 = (
            l_self_modules_blocks_modules_22_modules_1_parameters_weight_
        ) = l_self_modules_blocks_modules_22_modules_1_parameters_bias_ = None
        input_163 = torch.nn.functional.relu(input_162, inplace=False)
        input_162 = None
        input_164 = torch.nn.functional.batch_norm(
            input_163,
            l_self_modules_blocks_modules_22_modules_3_buffers_running_mean_,
            l_self_modules_blocks_modules_22_modules_3_buffers_running_var_,
            l_self_modules_blocks_modules_22_modules_3_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_163 = (
            l_self_modules_blocks_modules_22_modules_3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_22_modules_3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_22_modules_3_parameters_weight_
        ) = l_self_modules_blocks_modules_22_modules_3_parameters_bias_ = None
        input_165 = torch.conv2d(
            input_164,
            l_self_modules_blocks_modules_23_modules_0_modules_fn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_0_modules_fn_modules_0_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            768,
        )
        l_self_modules_blocks_modules_23_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_23_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_166 = torch.nn.functional.relu(input_165, inplace=False)
        input_165 = None
        input_167 = torch.nn.functional.batch_norm(
            input_166,
            l_self_modules_blocks_modules_23_modules_0_modules_fn_modules_2_buffers_running_mean_,
            l_self_modules_blocks_modules_23_modules_0_modules_fn_modules_2_buffers_running_var_,
            l_self_modules_blocks_modules_23_modules_0_modules_fn_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_0_modules_fn_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_166 = l_self_modules_blocks_modules_23_modules_0_modules_fn_modules_2_buffers_running_mean_ = l_self_modules_blocks_modules_23_modules_0_modules_fn_modules_2_buffers_running_var_ = l_self_modules_blocks_modules_23_modules_0_modules_fn_modules_2_parameters_weight_ = l_self_modules_blocks_modules_23_modules_0_modules_fn_modules_2_parameters_bias_ = (None)
        input_168 = input_167 + input_164
        input_167 = input_164 = None
        input_169 = torch.conv2d(
            input_168,
            l_self_modules_blocks_modules_23_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_168 = (
            l_self_modules_blocks_modules_23_modules_1_parameters_weight_
        ) = l_self_modules_blocks_modules_23_modules_1_parameters_bias_ = None
        input_170 = torch.nn.functional.relu(input_169, inplace=False)
        input_169 = None
        input_171 = torch.nn.functional.batch_norm(
            input_170,
            l_self_modules_blocks_modules_23_modules_3_buffers_running_mean_,
            l_self_modules_blocks_modules_23_modules_3_buffers_running_var_,
            l_self_modules_blocks_modules_23_modules_3_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_170 = (
            l_self_modules_blocks_modules_23_modules_3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_23_modules_3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_23_modules_3_parameters_weight_
        ) = l_self_modules_blocks_modules_23_modules_3_parameters_bias_ = None
        input_172 = torch.conv2d(
            input_171,
            l_self_modules_blocks_modules_24_modules_0_modules_fn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_24_modules_0_modules_fn_modules_0_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            768,
        )
        l_self_modules_blocks_modules_24_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_24_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_173 = torch.nn.functional.relu(input_172, inplace=False)
        input_172 = None
        input_174 = torch.nn.functional.batch_norm(
            input_173,
            l_self_modules_blocks_modules_24_modules_0_modules_fn_modules_2_buffers_running_mean_,
            l_self_modules_blocks_modules_24_modules_0_modules_fn_modules_2_buffers_running_var_,
            l_self_modules_blocks_modules_24_modules_0_modules_fn_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_24_modules_0_modules_fn_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_173 = l_self_modules_blocks_modules_24_modules_0_modules_fn_modules_2_buffers_running_mean_ = l_self_modules_blocks_modules_24_modules_0_modules_fn_modules_2_buffers_running_var_ = l_self_modules_blocks_modules_24_modules_0_modules_fn_modules_2_parameters_weight_ = l_self_modules_blocks_modules_24_modules_0_modules_fn_modules_2_parameters_bias_ = (None)
        input_175 = input_174 + input_171
        input_174 = input_171 = None
        input_176 = torch.conv2d(
            input_175,
            l_self_modules_blocks_modules_24_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_24_modules_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_175 = (
            l_self_modules_blocks_modules_24_modules_1_parameters_weight_
        ) = l_self_modules_blocks_modules_24_modules_1_parameters_bias_ = None
        input_177 = torch.nn.functional.relu(input_176, inplace=False)
        input_176 = None
        input_178 = torch.nn.functional.batch_norm(
            input_177,
            l_self_modules_blocks_modules_24_modules_3_buffers_running_mean_,
            l_self_modules_blocks_modules_24_modules_3_buffers_running_var_,
            l_self_modules_blocks_modules_24_modules_3_parameters_weight_,
            l_self_modules_blocks_modules_24_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_177 = (
            l_self_modules_blocks_modules_24_modules_3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_24_modules_3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_24_modules_3_parameters_weight_
        ) = l_self_modules_blocks_modules_24_modules_3_parameters_bias_ = None
        input_179 = torch.conv2d(
            input_178,
            l_self_modules_blocks_modules_25_modules_0_modules_fn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_25_modules_0_modules_fn_modules_0_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            768,
        )
        l_self_modules_blocks_modules_25_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_25_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_180 = torch.nn.functional.relu(input_179, inplace=False)
        input_179 = None
        input_181 = torch.nn.functional.batch_norm(
            input_180,
            l_self_modules_blocks_modules_25_modules_0_modules_fn_modules_2_buffers_running_mean_,
            l_self_modules_blocks_modules_25_modules_0_modules_fn_modules_2_buffers_running_var_,
            l_self_modules_blocks_modules_25_modules_0_modules_fn_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_25_modules_0_modules_fn_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_180 = l_self_modules_blocks_modules_25_modules_0_modules_fn_modules_2_buffers_running_mean_ = l_self_modules_blocks_modules_25_modules_0_modules_fn_modules_2_buffers_running_var_ = l_self_modules_blocks_modules_25_modules_0_modules_fn_modules_2_parameters_weight_ = l_self_modules_blocks_modules_25_modules_0_modules_fn_modules_2_parameters_bias_ = (None)
        input_182 = input_181 + input_178
        input_181 = input_178 = None
        input_183 = torch.conv2d(
            input_182,
            l_self_modules_blocks_modules_25_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_25_modules_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_182 = (
            l_self_modules_blocks_modules_25_modules_1_parameters_weight_
        ) = l_self_modules_blocks_modules_25_modules_1_parameters_bias_ = None
        input_184 = torch.nn.functional.relu(input_183, inplace=False)
        input_183 = None
        input_185 = torch.nn.functional.batch_norm(
            input_184,
            l_self_modules_blocks_modules_25_modules_3_buffers_running_mean_,
            l_self_modules_blocks_modules_25_modules_3_buffers_running_var_,
            l_self_modules_blocks_modules_25_modules_3_parameters_weight_,
            l_self_modules_blocks_modules_25_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_184 = (
            l_self_modules_blocks_modules_25_modules_3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_25_modules_3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_25_modules_3_parameters_weight_
        ) = l_self_modules_blocks_modules_25_modules_3_parameters_bias_ = None
        input_186 = torch.conv2d(
            input_185,
            l_self_modules_blocks_modules_26_modules_0_modules_fn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_26_modules_0_modules_fn_modules_0_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            768,
        )
        l_self_modules_blocks_modules_26_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_26_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_187 = torch.nn.functional.relu(input_186, inplace=False)
        input_186 = None
        input_188 = torch.nn.functional.batch_norm(
            input_187,
            l_self_modules_blocks_modules_26_modules_0_modules_fn_modules_2_buffers_running_mean_,
            l_self_modules_blocks_modules_26_modules_0_modules_fn_modules_2_buffers_running_var_,
            l_self_modules_blocks_modules_26_modules_0_modules_fn_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_26_modules_0_modules_fn_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_187 = l_self_modules_blocks_modules_26_modules_0_modules_fn_modules_2_buffers_running_mean_ = l_self_modules_blocks_modules_26_modules_0_modules_fn_modules_2_buffers_running_var_ = l_self_modules_blocks_modules_26_modules_0_modules_fn_modules_2_parameters_weight_ = l_self_modules_blocks_modules_26_modules_0_modules_fn_modules_2_parameters_bias_ = (None)
        input_189 = input_188 + input_185
        input_188 = input_185 = None
        input_190 = torch.conv2d(
            input_189,
            l_self_modules_blocks_modules_26_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_26_modules_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_189 = (
            l_self_modules_blocks_modules_26_modules_1_parameters_weight_
        ) = l_self_modules_blocks_modules_26_modules_1_parameters_bias_ = None
        input_191 = torch.nn.functional.relu(input_190, inplace=False)
        input_190 = None
        input_192 = torch.nn.functional.batch_norm(
            input_191,
            l_self_modules_blocks_modules_26_modules_3_buffers_running_mean_,
            l_self_modules_blocks_modules_26_modules_3_buffers_running_var_,
            l_self_modules_blocks_modules_26_modules_3_parameters_weight_,
            l_self_modules_blocks_modules_26_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_191 = (
            l_self_modules_blocks_modules_26_modules_3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_26_modules_3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_26_modules_3_parameters_weight_
        ) = l_self_modules_blocks_modules_26_modules_3_parameters_bias_ = None
        input_193 = torch.conv2d(
            input_192,
            l_self_modules_blocks_modules_27_modules_0_modules_fn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_27_modules_0_modules_fn_modules_0_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            768,
        )
        l_self_modules_blocks_modules_27_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_27_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_194 = torch.nn.functional.relu(input_193, inplace=False)
        input_193 = None
        input_195 = torch.nn.functional.batch_norm(
            input_194,
            l_self_modules_blocks_modules_27_modules_0_modules_fn_modules_2_buffers_running_mean_,
            l_self_modules_blocks_modules_27_modules_0_modules_fn_modules_2_buffers_running_var_,
            l_self_modules_blocks_modules_27_modules_0_modules_fn_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_27_modules_0_modules_fn_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_194 = l_self_modules_blocks_modules_27_modules_0_modules_fn_modules_2_buffers_running_mean_ = l_self_modules_blocks_modules_27_modules_0_modules_fn_modules_2_buffers_running_var_ = l_self_modules_blocks_modules_27_modules_0_modules_fn_modules_2_parameters_weight_ = l_self_modules_blocks_modules_27_modules_0_modules_fn_modules_2_parameters_bias_ = (None)
        input_196 = input_195 + input_192
        input_195 = input_192 = None
        input_197 = torch.conv2d(
            input_196,
            l_self_modules_blocks_modules_27_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_27_modules_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_196 = (
            l_self_modules_blocks_modules_27_modules_1_parameters_weight_
        ) = l_self_modules_blocks_modules_27_modules_1_parameters_bias_ = None
        input_198 = torch.nn.functional.relu(input_197, inplace=False)
        input_197 = None
        input_199 = torch.nn.functional.batch_norm(
            input_198,
            l_self_modules_blocks_modules_27_modules_3_buffers_running_mean_,
            l_self_modules_blocks_modules_27_modules_3_buffers_running_var_,
            l_self_modules_blocks_modules_27_modules_3_parameters_weight_,
            l_self_modules_blocks_modules_27_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_198 = (
            l_self_modules_blocks_modules_27_modules_3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_27_modules_3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_27_modules_3_parameters_weight_
        ) = l_self_modules_blocks_modules_27_modules_3_parameters_bias_ = None
        input_200 = torch.conv2d(
            input_199,
            l_self_modules_blocks_modules_28_modules_0_modules_fn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_28_modules_0_modules_fn_modules_0_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            768,
        )
        l_self_modules_blocks_modules_28_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_28_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_201 = torch.nn.functional.relu(input_200, inplace=False)
        input_200 = None
        input_202 = torch.nn.functional.batch_norm(
            input_201,
            l_self_modules_blocks_modules_28_modules_0_modules_fn_modules_2_buffers_running_mean_,
            l_self_modules_blocks_modules_28_modules_0_modules_fn_modules_2_buffers_running_var_,
            l_self_modules_blocks_modules_28_modules_0_modules_fn_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_28_modules_0_modules_fn_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_201 = l_self_modules_blocks_modules_28_modules_0_modules_fn_modules_2_buffers_running_mean_ = l_self_modules_blocks_modules_28_modules_0_modules_fn_modules_2_buffers_running_var_ = l_self_modules_blocks_modules_28_modules_0_modules_fn_modules_2_parameters_weight_ = l_self_modules_blocks_modules_28_modules_0_modules_fn_modules_2_parameters_bias_ = (None)
        input_203 = input_202 + input_199
        input_202 = input_199 = None
        input_204 = torch.conv2d(
            input_203,
            l_self_modules_blocks_modules_28_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_28_modules_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_203 = (
            l_self_modules_blocks_modules_28_modules_1_parameters_weight_
        ) = l_self_modules_blocks_modules_28_modules_1_parameters_bias_ = None
        input_205 = torch.nn.functional.relu(input_204, inplace=False)
        input_204 = None
        input_206 = torch.nn.functional.batch_norm(
            input_205,
            l_self_modules_blocks_modules_28_modules_3_buffers_running_mean_,
            l_self_modules_blocks_modules_28_modules_3_buffers_running_var_,
            l_self_modules_blocks_modules_28_modules_3_parameters_weight_,
            l_self_modules_blocks_modules_28_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_205 = (
            l_self_modules_blocks_modules_28_modules_3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_28_modules_3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_28_modules_3_parameters_weight_
        ) = l_self_modules_blocks_modules_28_modules_3_parameters_bias_ = None
        input_207 = torch.conv2d(
            input_206,
            l_self_modules_blocks_modules_29_modules_0_modules_fn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_29_modules_0_modules_fn_modules_0_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            768,
        )
        l_self_modules_blocks_modules_29_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_29_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_208 = torch.nn.functional.relu(input_207, inplace=False)
        input_207 = None
        input_209 = torch.nn.functional.batch_norm(
            input_208,
            l_self_modules_blocks_modules_29_modules_0_modules_fn_modules_2_buffers_running_mean_,
            l_self_modules_blocks_modules_29_modules_0_modules_fn_modules_2_buffers_running_var_,
            l_self_modules_blocks_modules_29_modules_0_modules_fn_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_29_modules_0_modules_fn_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_208 = l_self_modules_blocks_modules_29_modules_0_modules_fn_modules_2_buffers_running_mean_ = l_self_modules_blocks_modules_29_modules_0_modules_fn_modules_2_buffers_running_var_ = l_self_modules_blocks_modules_29_modules_0_modules_fn_modules_2_parameters_weight_ = l_self_modules_blocks_modules_29_modules_0_modules_fn_modules_2_parameters_bias_ = (None)
        input_210 = input_209 + input_206
        input_209 = input_206 = None
        input_211 = torch.conv2d(
            input_210,
            l_self_modules_blocks_modules_29_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_29_modules_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_210 = (
            l_self_modules_blocks_modules_29_modules_1_parameters_weight_
        ) = l_self_modules_blocks_modules_29_modules_1_parameters_bias_ = None
        input_212 = torch.nn.functional.relu(input_211, inplace=False)
        input_211 = None
        input_213 = torch.nn.functional.batch_norm(
            input_212,
            l_self_modules_blocks_modules_29_modules_3_buffers_running_mean_,
            l_self_modules_blocks_modules_29_modules_3_buffers_running_var_,
            l_self_modules_blocks_modules_29_modules_3_parameters_weight_,
            l_self_modules_blocks_modules_29_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_212 = (
            l_self_modules_blocks_modules_29_modules_3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_29_modules_3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_29_modules_3_parameters_weight_
        ) = l_self_modules_blocks_modules_29_modules_3_parameters_bias_ = None
        input_214 = torch.conv2d(
            input_213,
            l_self_modules_blocks_modules_30_modules_0_modules_fn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_30_modules_0_modules_fn_modules_0_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            768,
        )
        l_self_modules_blocks_modules_30_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_30_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_215 = torch.nn.functional.relu(input_214, inplace=False)
        input_214 = None
        input_216 = torch.nn.functional.batch_norm(
            input_215,
            l_self_modules_blocks_modules_30_modules_0_modules_fn_modules_2_buffers_running_mean_,
            l_self_modules_blocks_modules_30_modules_0_modules_fn_modules_2_buffers_running_var_,
            l_self_modules_blocks_modules_30_modules_0_modules_fn_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_30_modules_0_modules_fn_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_215 = l_self_modules_blocks_modules_30_modules_0_modules_fn_modules_2_buffers_running_mean_ = l_self_modules_blocks_modules_30_modules_0_modules_fn_modules_2_buffers_running_var_ = l_self_modules_blocks_modules_30_modules_0_modules_fn_modules_2_parameters_weight_ = l_self_modules_blocks_modules_30_modules_0_modules_fn_modules_2_parameters_bias_ = (None)
        input_217 = input_216 + input_213
        input_216 = input_213 = None
        input_218 = torch.conv2d(
            input_217,
            l_self_modules_blocks_modules_30_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_30_modules_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_217 = (
            l_self_modules_blocks_modules_30_modules_1_parameters_weight_
        ) = l_self_modules_blocks_modules_30_modules_1_parameters_bias_ = None
        input_219 = torch.nn.functional.relu(input_218, inplace=False)
        input_218 = None
        input_220 = torch.nn.functional.batch_norm(
            input_219,
            l_self_modules_blocks_modules_30_modules_3_buffers_running_mean_,
            l_self_modules_blocks_modules_30_modules_3_buffers_running_var_,
            l_self_modules_blocks_modules_30_modules_3_parameters_weight_,
            l_self_modules_blocks_modules_30_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_219 = (
            l_self_modules_blocks_modules_30_modules_3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_30_modules_3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_30_modules_3_parameters_weight_
        ) = l_self_modules_blocks_modules_30_modules_3_parameters_bias_ = None
        input_221 = torch.conv2d(
            input_220,
            l_self_modules_blocks_modules_31_modules_0_modules_fn_modules_0_parameters_weight_,
            l_self_modules_blocks_modules_31_modules_0_modules_fn_modules_0_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            768,
        )
        l_self_modules_blocks_modules_31_modules_0_modules_fn_modules_0_parameters_weight_ = l_self_modules_blocks_modules_31_modules_0_modules_fn_modules_0_parameters_bias_ = (None)
        input_222 = torch.nn.functional.relu(input_221, inplace=False)
        input_221 = None
        input_223 = torch.nn.functional.batch_norm(
            input_222,
            l_self_modules_blocks_modules_31_modules_0_modules_fn_modules_2_buffers_running_mean_,
            l_self_modules_blocks_modules_31_modules_0_modules_fn_modules_2_buffers_running_var_,
            l_self_modules_blocks_modules_31_modules_0_modules_fn_modules_2_parameters_weight_,
            l_self_modules_blocks_modules_31_modules_0_modules_fn_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_222 = l_self_modules_blocks_modules_31_modules_0_modules_fn_modules_2_buffers_running_mean_ = l_self_modules_blocks_modules_31_modules_0_modules_fn_modules_2_buffers_running_var_ = l_self_modules_blocks_modules_31_modules_0_modules_fn_modules_2_parameters_weight_ = l_self_modules_blocks_modules_31_modules_0_modules_fn_modules_2_parameters_bias_ = (None)
        input_224 = input_223 + input_220
        input_223 = input_220 = None
        input_225 = torch.conv2d(
            input_224,
            l_self_modules_blocks_modules_31_modules_1_parameters_weight_,
            l_self_modules_blocks_modules_31_modules_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_224 = (
            l_self_modules_blocks_modules_31_modules_1_parameters_weight_
        ) = l_self_modules_blocks_modules_31_modules_1_parameters_bias_ = None
        input_226 = torch.nn.functional.relu(input_225, inplace=False)
        input_225 = None
        input_227 = torch.nn.functional.batch_norm(
            input_226,
            l_self_modules_blocks_modules_31_modules_3_buffers_running_mean_,
            l_self_modules_blocks_modules_31_modules_3_buffers_running_var_,
            l_self_modules_blocks_modules_31_modules_3_parameters_weight_,
            l_self_modules_blocks_modules_31_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_226 = (
            l_self_modules_blocks_modules_31_modules_3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_31_modules_3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_31_modules_3_parameters_weight_
        ) = l_self_modules_blocks_modules_31_modules_3_parameters_bias_ = None
        x = torch.nn.functional.adaptive_avg_pool2d(input_227, 1)
        input_227 = None
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
