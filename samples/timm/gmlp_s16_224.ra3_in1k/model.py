import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_x_: torch.Tensor,
        L_self_modules_stem_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_24_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_24_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_24_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_24_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_24_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_24_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_24_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_24_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_24_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_24_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_25_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_25_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_25_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_25_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_25_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_25_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_25_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_25_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_25_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_25_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_26_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_26_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_26_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_26_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_26_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_26_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_26_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_26_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_26_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_26_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_27_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_27_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_27_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_27_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_27_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_27_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_27_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_27_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_27_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_27_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_28_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_28_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_28_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_28_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_28_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_28_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_28_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_28_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_28_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_28_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_29_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_29_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_29_modules_mlp_channels_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_29_modules_mlp_channels_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_29_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_29_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_29_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_29_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_29_modules_mlp_channels_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_29_modules_mlp_channels_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_x_ = L_x_
        l_self_modules_stem_modules_proj_parameters_weight_ = (
            L_self_modules_stem_modules_proj_parameters_weight_
        )
        l_self_modules_stem_modules_proj_parameters_bias_ = (
            L_self_modules_stem_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_0_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_0_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_0_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_0_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_0_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_0_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_0_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_0_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_0_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_0_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_1_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_1_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_1_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_1_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_1_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_1_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_1_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_1_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_1_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_2_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_2_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_2_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_2_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_2_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_2_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_2_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_2_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_2_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_3_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_3_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_3_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_3_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_3_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_3_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_3_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_3_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_3_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_4_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_4_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_4_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_4_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_4_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_4_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_4_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_4_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_4_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_5_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_5_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_5_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_5_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_5_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_5_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_5_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_5_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_5_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_6_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_6_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_6_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_6_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_6_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_6_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_6_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_6_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_6_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_6_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_7_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_7_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_7_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_7_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_7_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_7_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_7_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_7_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_7_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_7_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_7_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_7_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_8_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_8_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_8_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_8_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_8_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_8_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_8_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_8_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_8_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_8_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_8_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_8_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_9_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_9_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_9_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_9_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_9_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_9_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_9_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_9_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_9_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_9_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_9_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_9_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_10_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_10_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_10_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_10_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_10_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_10_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_10_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_10_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_10_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_10_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_10_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_10_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_11_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_11_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_11_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_11_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_11_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_11_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_11_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_11_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_11_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_11_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_11_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_11_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_12_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_12_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_12_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_12_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_12_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_12_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_12_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_12_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_12_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_12_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_12_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_12_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_13_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_13_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_13_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_13_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_13_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_13_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_13_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_13_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_13_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_13_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_13_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_13_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_14_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_14_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_14_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_14_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_14_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_14_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_14_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_14_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_14_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_14_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_14_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_14_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_15_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_15_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_15_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_15_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_15_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_15_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_15_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_15_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_15_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_15_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_15_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_15_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_16_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_16_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_16_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_16_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_16_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_16_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_16_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_16_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_16_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_16_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_16_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_16_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_17_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_17_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_17_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_17_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_17_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_17_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_17_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_17_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_17_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_17_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_17_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_17_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_18_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_18_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_18_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_18_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_18_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_18_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_18_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_18_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_18_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_18_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_18_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_18_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_19_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_19_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_19_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_19_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_19_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_19_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_19_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_19_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_19_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_19_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_19_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_19_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_20_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_20_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_20_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_20_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_20_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_20_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_20_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_20_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_20_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_20_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_20_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_20_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_21_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_21_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_21_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_21_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_21_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_21_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_21_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_21_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_21_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_21_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_21_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_21_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_22_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_22_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_22_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_22_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_22_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_22_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_22_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_22_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_22_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_22_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_22_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_22_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_23_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_23_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_23_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_23_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_23_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_23_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_23_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_23_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_23_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_23_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_23_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_23_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_24_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_24_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_24_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_24_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_24_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_24_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_24_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_24_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_24_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_24_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_24_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_24_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_24_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_24_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_24_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_24_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_24_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_24_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_24_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_24_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_25_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_25_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_25_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_25_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_25_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_25_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_25_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_25_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_25_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_25_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_25_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_25_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_25_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_25_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_25_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_25_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_25_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_25_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_25_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_25_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_26_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_26_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_26_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_26_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_26_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_26_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_26_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_26_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_26_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_26_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_26_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_26_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_26_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_26_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_26_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_26_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_26_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_26_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_26_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_26_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_27_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_27_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_27_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_27_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_27_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_27_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_27_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_27_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_27_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_27_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_27_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_27_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_27_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_27_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_27_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_27_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_27_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_27_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_27_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_27_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_28_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_28_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_28_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_28_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_28_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_28_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_28_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_28_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_28_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_28_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_28_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_28_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_28_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_28_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_28_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_28_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_28_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_28_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_28_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_28_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_blocks_modules_29_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_29_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_29_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_29_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_29_modules_mlp_channels_modules_fc1_parameters_weight_ = L_self_modules_blocks_modules_29_modules_mlp_channels_modules_fc1_parameters_weight_
        l_self_modules_blocks_modules_29_modules_mlp_channels_modules_fc1_parameters_bias_ = L_self_modules_blocks_modules_29_modules_mlp_channels_modules_fc1_parameters_bias_
        l_self_modules_blocks_modules_29_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_29_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_29_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_29_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_29_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_29_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_29_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = L_self_modules_blocks_modules_29_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_
        l_self_modules_blocks_modules_29_modules_mlp_channels_modules_fc2_parameters_weight_ = L_self_modules_blocks_modules_29_modules_mlp_channels_modules_fc2_parameters_weight_
        l_self_modules_blocks_modules_29_modules_mlp_channels_modules_fc2_parameters_bias_ = L_self_modules_blocks_modules_29_modules_mlp_channels_modules_fc2_parameters_bias_
        l_self_modules_norm_parameters_weight_ = L_self_modules_norm_parameters_weight_
        l_self_modules_norm_parameters_bias_ = L_self_modules_norm_parameters_bias_
        l_self_modules_head_parameters_weight_ = L_self_modules_head_parameters_weight_
        l_self_modules_head_parameters_bias_ = L_self_modules_head_parameters_bias_
        x = torch.conv2d(
            l_x_,
            l_self_modules_stem_modules_proj_parameters_weight_,
            l_self_modules_stem_modules_proj_parameters_bias_,
            (16, 16),
            (0, 0),
            (1, 1),
            1,
        )
        l_x_ = (
            l_self_modules_stem_modules_proj_parameters_weight_
        ) = l_self_modules_stem_modules_proj_parameters_bias_ = None
        flatten = x.flatten(2)
        x = None
        x_1 = flatten.transpose(1, 2)
        flatten = None
        layer_norm = torch.nn.functional.layer_norm(
            x_1,
            (256,),
            l_self_modules_blocks_modules_0_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_0_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_0_modules_norm_parameters_bias_
        ) = None
        x_2 = torch._C._nn.linear(
            layer_norm,
            l_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm = l_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_3 = torch._C._nn.gelu(x_2, approximate="none")
        x_2 = None
        x_4 = torch.nn.functional.dropout(x_3, 0.0, False, False)
        x_3 = None
        chunk = x_4.chunk(2, dim=-1)
        x_4 = None
        u = chunk[0]
        v = chunk[1]
        chunk = None
        v_1 = torch.nn.functional.layer_norm(
            v,
            (768,),
            l_self_modules_blocks_modules_0_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_,
            1e-05,
        )
        v = l_self_modules_blocks_modules_0_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_0_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = (None)
        transpose_1 = v_1.transpose(-1, -2)
        v_1 = None
        v_2 = torch._C._nn.linear(
            transpose_1,
            l_self_modules_blocks_modules_0_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_,
        )
        transpose_1 = l_self_modules_blocks_modules_0_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_0_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = (None)
        transpose_2 = v_2.transpose(-1, -2)
        v_2 = None
        x_5 = u * transpose_2
        u = transpose_2 = None
        x_6 = torch._C._nn.linear(
            x_5,
            l_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_5 = l_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_7 = torch.nn.functional.dropout(x_6, 0.0, False, False)
        x_6 = None
        x_8 = x_1 + x_7
        x_1 = x_7 = None
        layer_norm_2 = torch.nn.functional.layer_norm(
            x_8,
            (256,),
            l_self_modules_blocks_modules_1_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_1_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_1_modules_norm_parameters_bias_
        ) = None
        x_9 = torch._C._nn.linear(
            layer_norm_2,
            l_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_2 = l_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_10 = torch._C._nn.gelu(x_9, approximate="none")
        x_9 = None
        x_11 = torch.nn.functional.dropout(x_10, 0.0, False, False)
        x_10 = None
        chunk_1 = x_11.chunk(2, dim=-1)
        x_11 = None
        u_1 = chunk_1[0]
        v_3 = chunk_1[1]
        chunk_1 = None
        v_4 = torch.nn.functional.layer_norm(
            v_3,
            (768,),
            l_self_modules_blocks_modules_1_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_,
            1e-05,
        )
        v_3 = l_self_modules_blocks_modules_1_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_1_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = (None)
        transpose_3 = v_4.transpose(-1, -2)
        v_4 = None
        v_5 = torch._C._nn.linear(
            transpose_3,
            l_self_modules_blocks_modules_1_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_,
        )
        transpose_3 = l_self_modules_blocks_modules_1_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_1_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = (None)
        transpose_4 = v_5.transpose(-1, -2)
        v_5 = None
        x_12 = u_1 * transpose_4
        u_1 = transpose_4 = None
        x_13 = torch._C._nn.linear(
            x_12,
            l_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_12 = l_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_14 = torch.nn.functional.dropout(x_13, 0.0, False, False)
        x_13 = None
        x_15 = x_8 + x_14
        x_8 = x_14 = None
        layer_norm_4 = torch.nn.functional.layer_norm(
            x_15,
            (256,),
            l_self_modules_blocks_modules_2_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_2_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_2_modules_norm_parameters_bias_
        ) = None
        x_16 = torch._C._nn.linear(
            layer_norm_4,
            l_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_4 = l_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_17 = torch._C._nn.gelu(x_16, approximate="none")
        x_16 = None
        x_18 = torch.nn.functional.dropout(x_17, 0.0, False, False)
        x_17 = None
        chunk_2 = x_18.chunk(2, dim=-1)
        x_18 = None
        u_2 = chunk_2[0]
        v_6 = chunk_2[1]
        chunk_2 = None
        v_7 = torch.nn.functional.layer_norm(
            v_6,
            (768,),
            l_self_modules_blocks_modules_2_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_,
            1e-05,
        )
        v_6 = l_self_modules_blocks_modules_2_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_2_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = (None)
        transpose_5 = v_7.transpose(-1, -2)
        v_7 = None
        v_8 = torch._C._nn.linear(
            transpose_5,
            l_self_modules_blocks_modules_2_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_,
        )
        transpose_5 = l_self_modules_blocks_modules_2_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_2_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = (None)
        transpose_6 = v_8.transpose(-1, -2)
        v_8 = None
        x_19 = u_2 * transpose_6
        u_2 = transpose_6 = None
        x_20 = torch._C._nn.linear(
            x_19,
            l_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_19 = l_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_21 = torch.nn.functional.dropout(x_20, 0.0, False, False)
        x_20 = None
        x_22 = x_15 + x_21
        x_15 = x_21 = None
        layer_norm_6 = torch.nn.functional.layer_norm(
            x_22,
            (256,),
            l_self_modules_blocks_modules_3_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_3_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_3_modules_norm_parameters_bias_
        ) = None
        x_23 = torch._C._nn.linear(
            layer_norm_6,
            l_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_6 = l_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_24 = torch._C._nn.gelu(x_23, approximate="none")
        x_23 = None
        x_25 = torch.nn.functional.dropout(x_24, 0.0, False, False)
        x_24 = None
        chunk_3 = x_25.chunk(2, dim=-1)
        x_25 = None
        u_3 = chunk_3[0]
        v_9 = chunk_3[1]
        chunk_3 = None
        v_10 = torch.nn.functional.layer_norm(
            v_9,
            (768,),
            l_self_modules_blocks_modules_3_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_,
            1e-05,
        )
        v_9 = l_self_modules_blocks_modules_3_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_3_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = (None)
        transpose_7 = v_10.transpose(-1, -2)
        v_10 = None
        v_11 = torch._C._nn.linear(
            transpose_7,
            l_self_modules_blocks_modules_3_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_,
        )
        transpose_7 = l_self_modules_blocks_modules_3_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_3_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = (None)
        transpose_8 = v_11.transpose(-1, -2)
        v_11 = None
        x_26 = u_3 * transpose_8
        u_3 = transpose_8 = None
        x_27 = torch._C._nn.linear(
            x_26,
            l_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_26 = l_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_28 = torch.nn.functional.dropout(x_27, 0.0, False, False)
        x_27 = None
        x_29 = x_22 + x_28
        x_22 = x_28 = None
        layer_norm_8 = torch.nn.functional.layer_norm(
            x_29,
            (256,),
            l_self_modules_blocks_modules_4_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_4_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_4_modules_norm_parameters_bias_
        ) = None
        x_30 = torch._C._nn.linear(
            layer_norm_8,
            l_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_8 = l_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_31 = torch._C._nn.gelu(x_30, approximate="none")
        x_30 = None
        x_32 = torch.nn.functional.dropout(x_31, 0.0, False, False)
        x_31 = None
        chunk_4 = x_32.chunk(2, dim=-1)
        x_32 = None
        u_4 = chunk_4[0]
        v_12 = chunk_4[1]
        chunk_4 = None
        v_13 = torch.nn.functional.layer_norm(
            v_12,
            (768,),
            l_self_modules_blocks_modules_4_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_,
            1e-05,
        )
        v_12 = l_self_modules_blocks_modules_4_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_4_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = (None)
        transpose_9 = v_13.transpose(-1, -2)
        v_13 = None
        v_14 = torch._C._nn.linear(
            transpose_9,
            l_self_modules_blocks_modules_4_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_,
        )
        transpose_9 = l_self_modules_blocks_modules_4_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_4_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = (None)
        transpose_10 = v_14.transpose(-1, -2)
        v_14 = None
        x_33 = u_4 * transpose_10
        u_4 = transpose_10 = None
        x_34 = torch._C._nn.linear(
            x_33,
            l_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_33 = l_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_35 = torch.nn.functional.dropout(x_34, 0.0, False, False)
        x_34 = None
        x_36 = x_29 + x_35
        x_29 = x_35 = None
        layer_norm_10 = torch.nn.functional.layer_norm(
            x_36,
            (256,),
            l_self_modules_blocks_modules_5_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_5_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_5_modules_norm_parameters_bias_
        ) = None
        x_37 = torch._C._nn.linear(
            layer_norm_10,
            l_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_10 = l_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_38 = torch._C._nn.gelu(x_37, approximate="none")
        x_37 = None
        x_39 = torch.nn.functional.dropout(x_38, 0.0, False, False)
        x_38 = None
        chunk_5 = x_39.chunk(2, dim=-1)
        x_39 = None
        u_5 = chunk_5[0]
        v_15 = chunk_5[1]
        chunk_5 = None
        v_16 = torch.nn.functional.layer_norm(
            v_15,
            (768,),
            l_self_modules_blocks_modules_5_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_,
            1e-05,
        )
        v_15 = l_self_modules_blocks_modules_5_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_5_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = (None)
        transpose_11 = v_16.transpose(-1, -2)
        v_16 = None
        v_17 = torch._C._nn.linear(
            transpose_11,
            l_self_modules_blocks_modules_5_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_,
        )
        transpose_11 = l_self_modules_blocks_modules_5_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_5_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = (None)
        transpose_12 = v_17.transpose(-1, -2)
        v_17 = None
        x_40 = u_5 * transpose_12
        u_5 = transpose_12 = None
        x_41 = torch._C._nn.linear(
            x_40,
            l_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_40 = l_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_42 = torch.nn.functional.dropout(x_41, 0.0, False, False)
        x_41 = None
        x_43 = x_36 + x_42
        x_36 = x_42 = None
        layer_norm_12 = torch.nn.functional.layer_norm(
            x_43,
            (256,),
            l_self_modules_blocks_modules_6_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_6_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_6_modules_norm_parameters_bias_
        ) = None
        x_44 = torch._C._nn.linear(
            layer_norm_12,
            l_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_12 = l_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_45 = torch._C._nn.gelu(x_44, approximate="none")
        x_44 = None
        x_46 = torch.nn.functional.dropout(x_45, 0.0, False, False)
        x_45 = None
        chunk_6 = x_46.chunk(2, dim=-1)
        x_46 = None
        u_6 = chunk_6[0]
        v_18 = chunk_6[1]
        chunk_6 = None
        v_19 = torch.nn.functional.layer_norm(
            v_18,
            (768,),
            l_self_modules_blocks_modules_6_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_,
            1e-05,
        )
        v_18 = l_self_modules_blocks_modules_6_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_6_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = (None)
        transpose_13 = v_19.transpose(-1, -2)
        v_19 = None
        v_20 = torch._C._nn.linear(
            transpose_13,
            l_self_modules_blocks_modules_6_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_,
        )
        transpose_13 = l_self_modules_blocks_modules_6_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_6_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = (None)
        transpose_14 = v_20.transpose(-1, -2)
        v_20 = None
        x_47 = u_6 * transpose_14
        u_6 = transpose_14 = None
        x_48 = torch._C._nn.linear(
            x_47,
            l_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_47 = l_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_49 = torch.nn.functional.dropout(x_48, 0.0, False, False)
        x_48 = None
        x_50 = x_43 + x_49
        x_43 = x_49 = None
        layer_norm_14 = torch.nn.functional.layer_norm(
            x_50,
            (256,),
            l_self_modules_blocks_modules_7_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_7_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_7_modules_norm_parameters_bias_
        ) = None
        x_51 = torch._C._nn.linear(
            layer_norm_14,
            l_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_14 = l_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_52 = torch._C._nn.gelu(x_51, approximate="none")
        x_51 = None
        x_53 = torch.nn.functional.dropout(x_52, 0.0, False, False)
        x_52 = None
        chunk_7 = x_53.chunk(2, dim=-1)
        x_53 = None
        u_7 = chunk_7[0]
        v_21 = chunk_7[1]
        chunk_7 = None
        v_22 = torch.nn.functional.layer_norm(
            v_21,
            (768,),
            l_self_modules_blocks_modules_7_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_,
            1e-05,
        )
        v_21 = l_self_modules_blocks_modules_7_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_7_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = (None)
        transpose_15 = v_22.transpose(-1, -2)
        v_22 = None
        v_23 = torch._C._nn.linear(
            transpose_15,
            l_self_modules_blocks_modules_7_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_,
        )
        transpose_15 = l_self_modules_blocks_modules_7_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_7_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = (None)
        transpose_16 = v_23.transpose(-1, -2)
        v_23 = None
        x_54 = u_7 * transpose_16
        u_7 = transpose_16 = None
        x_55 = torch._C._nn.linear(
            x_54,
            l_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_54 = l_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_56 = torch.nn.functional.dropout(x_55, 0.0, False, False)
        x_55 = None
        x_57 = x_50 + x_56
        x_50 = x_56 = None
        layer_norm_16 = torch.nn.functional.layer_norm(
            x_57,
            (256,),
            l_self_modules_blocks_modules_8_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_8_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_8_modules_norm_parameters_bias_
        ) = None
        x_58 = torch._C._nn.linear(
            layer_norm_16,
            l_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_16 = l_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_59 = torch._C._nn.gelu(x_58, approximate="none")
        x_58 = None
        x_60 = torch.nn.functional.dropout(x_59, 0.0, False, False)
        x_59 = None
        chunk_8 = x_60.chunk(2, dim=-1)
        x_60 = None
        u_8 = chunk_8[0]
        v_24 = chunk_8[1]
        chunk_8 = None
        v_25 = torch.nn.functional.layer_norm(
            v_24,
            (768,),
            l_self_modules_blocks_modules_8_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_,
            1e-05,
        )
        v_24 = l_self_modules_blocks_modules_8_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_8_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = (None)
        transpose_17 = v_25.transpose(-1, -2)
        v_25 = None
        v_26 = torch._C._nn.linear(
            transpose_17,
            l_self_modules_blocks_modules_8_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_,
        )
        transpose_17 = l_self_modules_blocks_modules_8_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_8_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = (None)
        transpose_18 = v_26.transpose(-1, -2)
        v_26 = None
        x_61 = u_8 * transpose_18
        u_8 = transpose_18 = None
        x_62 = torch._C._nn.linear(
            x_61,
            l_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_61 = l_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_63 = torch.nn.functional.dropout(x_62, 0.0, False, False)
        x_62 = None
        x_64 = x_57 + x_63
        x_57 = x_63 = None
        layer_norm_18 = torch.nn.functional.layer_norm(
            x_64,
            (256,),
            l_self_modules_blocks_modules_9_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_9_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_9_modules_norm_parameters_bias_
        ) = None
        x_65 = torch._C._nn.linear(
            layer_norm_18,
            l_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_18 = l_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_66 = torch._C._nn.gelu(x_65, approximate="none")
        x_65 = None
        x_67 = torch.nn.functional.dropout(x_66, 0.0, False, False)
        x_66 = None
        chunk_9 = x_67.chunk(2, dim=-1)
        x_67 = None
        u_9 = chunk_9[0]
        v_27 = chunk_9[1]
        chunk_9 = None
        v_28 = torch.nn.functional.layer_norm(
            v_27,
            (768,),
            l_self_modules_blocks_modules_9_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_,
            1e-05,
        )
        v_27 = l_self_modules_blocks_modules_9_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_9_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = (None)
        transpose_19 = v_28.transpose(-1, -2)
        v_28 = None
        v_29 = torch._C._nn.linear(
            transpose_19,
            l_self_modules_blocks_modules_9_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_,
        )
        transpose_19 = l_self_modules_blocks_modules_9_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_9_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = (None)
        transpose_20 = v_29.transpose(-1, -2)
        v_29 = None
        x_68 = u_9 * transpose_20
        u_9 = transpose_20 = None
        x_69 = torch._C._nn.linear(
            x_68,
            l_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_68 = l_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_70 = torch.nn.functional.dropout(x_69, 0.0, False, False)
        x_69 = None
        x_71 = x_64 + x_70
        x_64 = x_70 = None
        layer_norm_20 = torch.nn.functional.layer_norm(
            x_71,
            (256,),
            l_self_modules_blocks_modules_10_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_10_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_10_modules_norm_parameters_bias_
        ) = None
        x_72 = torch._C._nn.linear(
            layer_norm_20,
            l_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_20 = l_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_73 = torch._C._nn.gelu(x_72, approximate="none")
        x_72 = None
        x_74 = torch.nn.functional.dropout(x_73, 0.0, False, False)
        x_73 = None
        chunk_10 = x_74.chunk(2, dim=-1)
        x_74 = None
        u_10 = chunk_10[0]
        v_30 = chunk_10[1]
        chunk_10 = None
        v_31 = torch.nn.functional.layer_norm(
            v_30,
            (768,),
            l_self_modules_blocks_modules_10_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_,
            1e-05,
        )
        v_30 = l_self_modules_blocks_modules_10_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_10_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = (None)
        transpose_21 = v_31.transpose(-1, -2)
        v_31 = None
        v_32 = torch._C._nn.linear(
            transpose_21,
            l_self_modules_blocks_modules_10_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_,
        )
        transpose_21 = l_self_modules_blocks_modules_10_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_10_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = (None)
        transpose_22 = v_32.transpose(-1, -2)
        v_32 = None
        x_75 = u_10 * transpose_22
        u_10 = transpose_22 = None
        x_76 = torch._C._nn.linear(
            x_75,
            l_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_75 = l_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_77 = torch.nn.functional.dropout(x_76, 0.0, False, False)
        x_76 = None
        x_78 = x_71 + x_77
        x_71 = x_77 = None
        layer_norm_22 = torch.nn.functional.layer_norm(
            x_78,
            (256,),
            l_self_modules_blocks_modules_11_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_11_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_11_modules_norm_parameters_bias_
        ) = None
        x_79 = torch._C._nn.linear(
            layer_norm_22,
            l_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_22 = l_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_80 = torch._C._nn.gelu(x_79, approximate="none")
        x_79 = None
        x_81 = torch.nn.functional.dropout(x_80, 0.0, False, False)
        x_80 = None
        chunk_11 = x_81.chunk(2, dim=-1)
        x_81 = None
        u_11 = chunk_11[0]
        v_33 = chunk_11[1]
        chunk_11 = None
        v_34 = torch.nn.functional.layer_norm(
            v_33,
            (768,),
            l_self_modules_blocks_modules_11_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_,
            1e-05,
        )
        v_33 = l_self_modules_blocks_modules_11_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_11_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = (None)
        transpose_23 = v_34.transpose(-1, -2)
        v_34 = None
        v_35 = torch._C._nn.linear(
            transpose_23,
            l_self_modules_blocks_modules_11_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_,
        )
        transpose_23 = l_self_modules_blocks_modules_11_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_11_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = (None)
        transpose_24 = v_35.transpose(-1, -2)
        v_35 = None
        x_82 = u_11 * transpose_24
        u_11 = transpose_24 = None
        x_83 = torch._C._nn.linear(
            x_82,
            l_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_82 = l_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_84 = torch.nn.functional.dropout(x_83, 0.0, False, False)
        x_83 = None
        x_85 = x_78 + x_84
        x_78 = x_84 = None
        layer_norm_24 = torch.nn.functional.layer_norm(
            x_85,
            (256,),
            l_self_modules_blocks_modules_12_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_12_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_12_modules_norm_parameters_bias_
        ) = None
        x_86 = torch._C._nn.linear(
            layer_norm_24,
            l_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_24 = l_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_87 = torch._C._nn.gelu(x_86, approximate="none")
        x_86 = None
        x_88 = torch.nn.functional.dropout(x_87, 0.0, False, False)
        x_87 = None
        chunk_12 = x_88.chunk(2, dim=-1)
        x_88 = None
        u_12 = chunk_12[0]
        v_36 = chunk_12[1]
        chunk_12 = None
        v_37 = torch.nn.functional.layer_norm(
            v_36,
            (768,),
            l_self_modules_blocks_modules_12_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_,
            1e-05,
        )
        v_36 = l_self_modules_blocks_modules_12_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_12_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = (None)
        transpose_25 = v_37.transpose(-1, -2)
        v_37 = None
        v_38 = torch._C._nn.linear(
            transpose_25,
            l_self_modules_blocks_modules_12_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_,
        )
        transpose_25 = l_self_modules_blocks_modules_12_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_12_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = (None)
        transpose_26 = v_38.transpose(-1, -2)
        v_38 = None
        x_89 = u_12 * transpose_26
        u_12 = transpose_26 = None
        x_90 = torch._C._nn.linear(
            x_89,
            l_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_89 = l_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_12_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_91 = torch.nn.functional.dropout(x_90, 0.0, False, False)
        x_90 = None
        x_92 = x_85 + x_91
        x_85 = x_91 = None
        layer_norm_26 = torch.nn.functional.layer_norm(
            x_92,
            (256,),
            l_self_modules_blocks_modules_13_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_13_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_13_modules_norm_parameters_bias_
        ) = None
        x_93 = torch._C._nn.linear(
            layer_norm_26,
            l_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_26 = l_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_94 = torch._C._nn.gelu(x_93, approximate="none")
        x_93 = None
        x_95 = torch.nn.functional.dropout(x_94, 0.0, False, False)
        x_94 = None
        chunk_13 = x_95.chunk(2, dim=-1)
        x_95 = None
        u_13 = chunk_13[0]
        v_39 = chunk_13[1]
        chunk_13 = None
        v_40 = torch.nn.functional.layer_norm(
            v_39,
            (768,),
            l_self_modules_blocks_modules_13_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_,
            1e-05,
        )
        v_39 = l_self_modules_blocks_modules_13_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_13_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = (None)
        transpose_27 = v_40.transpose(-1, -2)
        v_40 = None
        v_41 = torch._C._nn.linear(
            transpose_27,
            l_self_modules_blocks_modules_13_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_,
        )
        transpose_27 = l_self_modules_blocks_modules_13_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_13_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = (None)
        transpose_28 = v_41.transpose(-1, -2)
        v_41 = None
        x_96 = u_13 * transpose_28
        u_13 = transpose_28 = None
        x_97 = torch._C._nn.linear(
            x_96,
            l_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_96 = l_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_13_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_98 = torch.nn.functional.dropout(x_97, 0.0, False, False)
        x_97 = None
        x_99 = x_92 + x_98
        x_92 = x_98 = None
        layer_norm_28 = torch.nn.functional.layer_norm(
            x_99,
            (256,),
            l_self_modules_blocks_modules_14_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_14_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_14_modules_norm_parameters_bias_
        ) = None
        x_100 = torch._C._nn.linear(
            layer_norm_28,
            l_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_28 = l_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_101 = torch._C._nn.gelu(x_100, approximate="none")
        x_100 = None
        x_102 = torch.nn.functional.dropout(x_101, 0.0, False, False)
        x_101 = None
        chunk_14 = x_102.chunk(2, dim=-1)
        x_102 = None
        u_14 = chunk_14[0]
        v_42 = chunk_14[1]
        chunk_14 = None
        v_43 = torch.nn.functional.layer_norm(
            v_42,
            (768,),
            l_self_modules_blocks_modules_14_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_,
            1e-05,
        )
        v_42 = l_self_modules_blocks_modules_14_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_14_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = (None)
        transpose_29 = v_43.transpose(-1, -2)
        v_43 = None
        v_44 = torch._C._nn.linear(
            transpose_29,
            l_self_modules_blocks_modules_14_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_,
        )
        transpose_29 = l_self_modules_blocks_modules_14_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_14_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = (None)
        transpose_30 = v_44.transpose(-1, -2)
        v_44 = None
        x_103 = u_14 * transpose_30
        u_14 = transpose_30 = None
        x_104 = torch._C._nn.linear(
            x_103,
            l_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_103 = l_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_14_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_105 = torch.nn.functional.dropout(x_104, 0.0, False, False)
        x_104 = None
        x_106 = x_99 + x_105
        x_99 = x_105 = None
        layer_norm_30 = torch.nn.functional.layer_norm(
            x_106,
            (256,),
            l_self_modules_blocks_modules_15_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_15_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_15_modules_norm_parameters_bias_
        ) = None
        x_107 = torch._C._nn.linear(
            layer_norm_30,
            l_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_30 = l_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_108 = torch._C._nn.gelu(x_107, approximate="none")
        x_107 = None
        x_109 = torch.nn.functional.dropout(x_108, 0.0, False, False)
        x_108 = None
        chunk_15 = x_109.chunk(2, dim=-1)
        x_109 = None
        u_15 = chunk_15[0]
        v_45 = chunk_15[1]
        chunk_15 = None
        v_46 = torch.nn.functional.layer_norm(
            v_45,
            (768,),
            l_self_modules_blocks_modules_15_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_,
            1e-05,
        )
        v_45 = l_self_modules_blocks_modules_15_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_15_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = (None)
        transpose_31 = v_46.transpose(-1, -2)
        v_46 = None
        v_47 = torch._C._nn.linear(
            transpose_31,
            l_self_modules_blocks_modules_15_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_,
        )
        transpose_31 = l_self_modules_blocks_modules_15_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_15_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = (None)
        transpose_32 = v_47.transpose(-1, -2)
        v_47 = None
        x_110 = u_15 * transpose_32
        u_15 = transpose_32 = None
        x_111 = torch._C._nn.linear(
            x_110,
            l_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_110 = l_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_15_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_112 = torch.nn.functional.dropout(x_111, 0.0, False, False)
        x_111 = None
        x_113 = x_106 + x_112
        x_106 = x_112 = None
        layer_norm_32 = torch.nn.functional.layer_norm(
            x_113,
            (256,),
            l_self_modules_blocks_modules_16_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_16_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_16_modules_norm_parameters_bias_
        ) = None
        x_114 = torch._C._nn.linear(
            layer_norm_32,
            l_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_32 = l_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_115 = torch._C._nn.gelu(x_114, approximate="none")
        x_114 = None
        x_116 = torch.nn.functional.dropout(x_115, 0.0, False, False)
        x_115 = None
        chunk_16 = x_116.chunk(2, dim=-1)
        x_116 = None
        u_16 = chunk_16[0]
        v_48 = chunk_16[1]
        chunk_16 = None
        v_49 = torch.nn.functional.layer_norm(
            v_48,
            (768,),
            l_self_modules_blocks_modules_16_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_,
            1e-05,
        )
        v_48 = l_self_modules_blocks_modules_16_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_16_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = (None)
        transpose_33 = v_49.transpose(-1, -2)
        v_49 = None
        v_50 = torch._C._nn.linear(
            transpose_33,
            l_self_modules_blocks_modules_16_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_,
        )
        transpose_33 = l_self_modules_blocks_modules_16_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_16_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = (None)
        transpose_34 = v_50.transpose(-1, -2)
        v_50 = None
        x_117 = u_16 * transpose_34
        u_16 = transpose_34 = None
        x_118 = torch._C._nn.linear(
            x_117,
            l_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_117 = l_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_16_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_119 = torch.nn.functional.dropout(x_118, 0.0, False, False)
        x_118 = None
        x_120 = x_113 + x_119
        x_113 = x_119 = None
        layer_norm_34 = torch.nn.functional.layer_norm(
            x_120,
            (256,),
            l_self_modules_blocks_modules_17_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_17_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_17_modules_norm_parameters_bias_
        ) = None
        x_121 = torch._C._nn.linear(
            layer_norm_34,
            l_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_34 = l_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_122 = torch._C._nn.gelu(x_121, approximate="none")
        x_121 = None
        x_123 = torch.nn.functional.dropout(x_122, 0.0, False, False)
        x_122 = None
        chunk_17 = x_123.chunk(2, dim=-1)
        x_123 = None
        u_17 = chunk_17[0]
        v_51 = chunk_17[1]
        chunk_17 = None
        v_52 = torch.nn.functional.layer_norm(
            v_51,
            (768,),
            l_self_modules_blocks_modules_17_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_,
            1e-05,
        )
        v_51 = l_self_modules_blocks_modules_17_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_17_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = (None)
        transpose_35 = v_52.transpose(-1, -2)
        v_52 = None
        v_53 = torch._C._nn.linear(
            transpose_35,
            l_self_modules_blocks_modules_17_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_,
        )
        transpose_35 = l_self_modules_blocks_modules_17_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_17_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = (None)
        transpose_36 = v_53.transpose(-1, -2)
        v_53 = None
        x_124 = u_17 * transpose_36
        u_17 = transpose_36 = None
        x_125 = torch._C._nn.linear(
            x_124,
            l_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_124 = l_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_17_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_126 = torch.nn.functional.dropout(x_125, 0.0, False, False)
        x_125 = None
        x_127 = x_120 + x_126
        x_120 = x_126 = None
        layer_norm_36 = torch.nn.functional.layer_norm(
            x_127,
            (256,),
            l_self_modules_blocks_modules_18_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_18_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_18_modules_norm_parameters_bias_
        ) = None
        x_128 = torch._C._nn.linear(
            layer_norm_36,
            l_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_36 = l_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_129 = torch._C._nn.gelu(x_128, approximate="none")
        x_128 = None
        x_130 = torch.nn.functional.dropout(x_129, 0.0, False, False)
        x_129 = None
        chunk_18 = x_130.chunk(2, dim=-1)
        x_130 = None
        u_18 = chunk_18[0]
        v_54 = chunk_18[1]
        chunk_18 = None
        v_55 = torch.nn.functional.layer_norm(
            v_54,
            (768,),
            l_self_modules_blocks_modules_18_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_,
            1e-05,
        )
        v_54 = l_self_modules_blocks_modules_18_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_18_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = (None)
        transpose_37 = v_55.transpose(-1, -2)
        v_55 = None
        v_56 = torch._C._nn.linear(
            transpose_37,
            l_self_modules_blocks_modules_18_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_,
        )
        transpose_37 = l_self_modules_blocks_modules_18_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_18_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = (None)
        transpose_38 = v_56.transpose(-1, -2)
        v_56 = None
        x_131 = u_18 * transpose_38
        u_18 = transpose_38 = None
        x_132 = torch._C._nn.linear(
            x_131,
            l_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_131 = l_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_18_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_133 = torch.nn.functional.dropout(x_132, 0.0, False, False)
        x_132 = None
        x_134 = x_127 + x_133
        x_127 = x_133 = None
        layer_norm_38 = torch.nn.functional.layer_norm(
            x_134,
            (256,),
            l_self_modules_blocks_modules_19_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_19_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_19_modules_norm_parameters_bias_
        ) = None
        x_135 = torch._C._nn.linear(
            layer_norm_38,
            l_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_38 = l_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_136 = torch._C._nn.gelu(x_135, approximate="none")
        x_135 = None
        x_137 = torch.nn.functional.dropout(x_136, 0.0, False, False)
        x_136 = None
        chunk_19 = x_137.chunk(2, dim=-1)
        x_137 = None
        u_19 = chunk_19[0]
        v_57 = chunk_19[1]
        chunk_19 = None
        v_58 = torch.nn.functional.layer_norm(
            v_57,
            (768,),
            l_self_modules_blocks_modules_19_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_,
            1e-05,
        )
        v_57 = l_self_modules_blocks_modules_19_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_19_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = (None)
        transpose_39 = v_58.transpose(-1, -2)
        v_58 = None
        v_59 = torch._C._nn.linear(
            transpose_39,
            l_self_modules_blocks_modules_19_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_,
        )
        transpose_39 = l_self_modules_blocks_modules_19_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_19_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = (None)
        transpose_40 = v_59.transpose(-1, -2)
        v_59 = None
        x_138 = u_19 * transpose_40
        u_19 = transpose_40 = None
        x_139 = torch._C._nn.linear(
            x_138,
            l_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_138 = l_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_19_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_140 = torch.nn.functional.dropout(x_139, 0.0, False, False)
        x_139 = None
        x_141 = x_134 + x_140
        x_134 = x_140 = None
        layer_norm_40 = torch.nn.functional.layer_norm(
            x_141,
            (256,),
            l_self_modules_blocks_modules_20_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_20_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_20_modules_norm_parameters_bias_
        ) = None
        x_142 = torch._C._nn.linear(
            layer_norm_40,
            l_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_40 = l_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_143 = torch._C._nn.gelu(x_142, approximate="none")
        x_142 = None
        x_144 = torch.nn.functional.dropout(x_143, 0.0, False, False)
        x_143 = None
        chunk_20 = x_144.chunk(2, dim=-1)
        x_144 = None
        u_20 = chunk_20[0]
        v_60 = chunk_20[1]
        chunk_20 = None
        v_61 = torch.nn.functional.layer_norm(
            v_60,
            (768,),
            l_self_modules_blocks_modules_20_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_,
            1e-05,
        )
        v_60 = l_self_modules_blocks_modules_20_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_20_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = (None)
        transpose_41 = v_61.transpose(-1, -2)
        v_61 = None
        v_62 = torch._C._nn.linear(
            transpose_41,
            l_self_modules_blocks_modules_20_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_,
        )
        transpose_41 = l_self_modules_blocks_modules_20_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_20_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = (None)
        transpose_42 = v_62.transpose(-1, -2)
        v_62 = None
        x_145 = u_20 * transpose_42
        u_20 = transpose_42 = None
        x_146 = torch._C._nn.linear(
            x_145,
            l_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_145 = l_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_20_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_147 = torch.nn.functional.dropout(x_146, 0.0, False, False)
        x_146 = None
        x_148 = x_141 + x_147
        x_141 = x_147 = None
        layer_norm_42 = torch.nn.functional.layer_norm(
            x_148,
            (256,),
            l_self_modules_blocks_modules_21_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_21_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_21_modules_norm_parameters_bias_
        ) = None
        x_149 = torch._C._nn.linear(
            layer_norm_42,
            l_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_42 = l_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_150 = torch._C._nn.gelu(x_149, approximate="none")
        x_149 = None
        x_151 = torch.nn.functional.dropout(x_150, 0.0, False, False)
        x_150 = None
        chunk_21 = x_151.chunk(2, dim=-1)
        x_151 = None
        u_21 = chunk_21[0]
        v_63 = chunk_21[1]
        chunk_21 = None
        v_64 = torch.nn.functional.layer_norm(
            v_63,
            (768,),
            l_self_modules_blocks_modules_21_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_,
            1e-05,
        )
        v_63 = l_self_modules_blocks_modules_21_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_21_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = (None)
        transpose_43 = v_64.transpose(-1, -2)
        v_64 = None
        v_65 = torch._C._nn.linear(
            transpose_43,
            l_self_modules_blocks_modules_21_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_,
        )
        transpose_43 = l_self_modules_blocks_modules_21_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_21_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = (None)
        transpose_44 = v_65.transpose(-1, -2)
        v_65 = None
        x_152 = u_21 * transpose_44
        u_21 = transpose_44 = None
        x_153 = torch._C._nn.linear(
            x_152,
            l_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_152 = l_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_21_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_154 = torch.nn.functional.dropout(x_153, 0.0, False, False)
        x_153 = None
        x_155 = x_148 + x_154
        x_148 = x_154 = None
        layer_norm_44 = torch.nn.functional.layer_norm(
            x_155,
            (256,),
            l_self_modules_blocks_modules_22_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_22_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_22_modules_norm_parameters_bias_
        ) = None
        x_156 = torch._C._nn.linear(
            layer_norm_44,
            l_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_44 = l_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_157 = torch._C._nn.gelu(x_156, approximate="none")
        x_156 = None
        x_158 = torch.nn.functional.dropout(x_157, 0.0, False, False)
        x_157 = None
        chunk_22 = x_158.chunk(2, dim=-1)
        x_158 = None
        u_22 = chunk_22[0]
        v_66 = chunk_22[1]
        chunk_22 = None
        v_67 = torch.nn.functional.layer_norm(
            v_66,
            (768,),
            l_self_modules_blocks_modules_22_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_,
            1e-05,
        )
        v_66 = l_self_modules_blocks_modules_22_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_22_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = (None)
        transpose_45 = v_67.transpose(-1, -2)
        v_67 = None
        v_68 = torch._C._nn.linear(
            transpose_45,
            l_self_modules_blocks_modules_22_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_,
        )
        transpose_45 = l_self_modules_blocks_modules_22_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_22_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = (None)
        transpose_46 = v_68.transpose(-1, -2)
        v_68 = None
        x_159 = u_22 * transpose_46
        u_22 = transpose_46 = None
        x_160 = torch._C._nn.linear(
            x_159,
            l_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_159 = l_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_22_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_161 = torch.nn.functional.dropout(x_160, 0.0, False, False)
        x_160 = None
        x_162 = x_155 + x_161
        x_155 = x_161 = None
        layer_norm_46 = torch.nn.functional.layer_norm(
            x_162,
            (256,),
            l_self_modules_blocks_modules_23_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_23_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_23_modules_norm_parameters_bias_
        ) = None
        x_163 = torch._C._nn.linear(
            layer_norm_46,
            l_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_46 = l_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_164 = torch._C._nn.gelu(x_163, approximate="none")
        x_163 = None
        x_165 = torch.nn.functional.dropout(x_164, 0.0, False, False)
        x_164 = None
        chunk_23 = x_165.chunk(2, dim=-1)
        x_165 = None
        u_23 = chunk_23[0]
        v_69 = chunk_23[1]
        chunk_23 = None
        v_70 = torch.nn.functional.layer_norm(
            v_69,
            (768,),
            l_self_modules_blocks_modules_23_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_,
            1e-05,
        )
        v_69 = l_self_modules_blocks_modules_23_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_23_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = (None)
        transpose_47 = v_70.transpose(-1, -2)
        v_70 = None
        v_71 = torch._C._nn.linear(
            transpose_47,
            l_self_modules_blocks_modules_23_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_,
        )
        transpose_47 = l_self_modules_blocks_modules_23_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_23_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = (None)
        transpose_48 = v_71.transpose(-1, -2)
        v_71 = None
        x_166 = u_23 * transpose_48
        u_23 = transpose_48 = None
        x_167 = torch._C._nn.linear(
            x_166,
            l_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_166 = l_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_23_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_168 = torch.nn.functional.dropout(x_167, 0.0, False, False)
        x_167 = None
        x_169 = x_162 + x_168
        x_162 = x_168 = None
        layer_norm_48 = torch.nn.functional.layer_norm(
            x_169,
            (256,),
            l_self_modules_blocks_modules_24_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_24_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_24_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_24_modules_norm_parameters_bias_
        ) = None
        x_170 = torch._C._nn.linear(
            layer_norm_48,
            l_self_modules_blocks_modules_24_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_24_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_48 = l_self_modules_blocks_modules_24_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_24_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_171 = torch._C._nn.gelu(x_170, approximate="none")
        x_170 = None
        x_172 = torch.nn.functional.dropout(x_171, 0.0, False, False)
        x_171 = None
        chunk_24 = x_172.chunk(2, dim=-1)
        x_172 = None
        u_24 = chunk_24[0]
        v_72 = chunk_24[1]
        chunk_24 = None
        v_73 = torch.nn.functional.layer_norm(
            v_72,
            (768,),
            l_self_modules_blocks_modules_24_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_24_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_,
            1e-05,
        )
        v_72 = l_self_modules_blocks_modules_24_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_24_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = (None)
        transpose_49 = v_73.transpose(-1, -2)
        v_73 = None
        v_74 = torch._C._nn.linear(
            transpose_49,
            l_self_modules_blocks_modules_24_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_24_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_,
        )
        transpose_49 = l_self_modules_blocks_modules_24_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_24_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = (None)
        transpose_50 = v_74.transpose(-1, -2)
        v_74 = None
        x_173 = u_24 * transpose_50
        u_24 = transpose_50 = None
        x_174 = torch._C._nn.linear(
            x_173,
            l_self_modules_blocks_modules_24_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_24_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_173 = l_self_modules_blocks_modules_24_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_24_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_175 = torch.nn.functional.dropout(x_174, 0.0, False, False)
        x_174 = None
        x_176 = x_169 + x_175
        x_169 = x_175 = None
        layer_norm_50 = torch.nn.functional.layer_norm(
            x_176,
            (256,),
            l_self_modules_blocks_modules_25_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_25_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_25_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_25_modules_norm_parameters_bias_
        ) = None
        x_177 = torch._C._nn.linear(
            layer_norm_50,
            l_self_modules_blocks_modules_25_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_25_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_50 = l_self_modules_blocks_modules_25_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_25_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_178 = torch._C._nn.gelu(x_177, approximate="none")
        x_177 = None
        x_179 = torch.nn.functional.dropout(x_178, 0.0, False, False)
        x_178 = None
        chunk_25 = x_179.chunk(2, dim=-1)
        x_179 = None
        u_25 = chunk_25[0]
        v_75 = chunk_25[1]
        chunk_25 = None
        v_76 = torch.nn.functional.layer_norm(
            v_75,
            (768,),
            l_self_modules_blocks_modules_25_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_25_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_,
            1e-05,
        )
        v_75 = l_self_modules_blocks_modules_25_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_25_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = (None)
        transpose_51 = v_76.transpose(-1, -2)
        v_76 = None
        v_77 = torch._C._nn.linear(
            transpose_51,
            l_self_modules_blocks_modules_25_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_25_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_,
        )
        transpose_51 = l_self_modules_blocks_modules_25_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_25_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = (None)
        transpose_52 = v_77.transpose(-1, -2)
        v_77 = None
        x_180 = u_25 * transpose_52
        u_25 = transpose_52 = None
        x_181 = torch._C._nn.linear(
            x_180,
            l_self_modules_blocks_modules_25_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_25_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_180 = l_self_modules_blocks_modules_25_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_25_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_182 = torch.nn.functional.dropout(x_181, 0.0, False, False)
        x_181 = None
        x_183 = x_176 + x_182
        x_176 = x_182 = None
        layer_norm_52 = torch.nn.functional.layer_norm(
            x_183,
            (256,),
            l_self_modules_blocks_modules_26_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_26_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_26_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_26_modules_norm_parameters_bias_
        ) = None
        x_184 = torch._C._nn.linear(
            layer_norm_52,
            l_self_modules_blocks_modules_26_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_26_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_52 = l_self_modules_blocks_modules_26_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_26_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_185 = torch._C._nn.gelu(x_184, approximate="none")
        x_184 = None
        x_186 = torch.nn.functional.dropout(x_185, 0.0, False, False)
        x_185 = None
        chunk_26 = x_186.chunk(2, dim=-1)
        x_186 = None
        u_26 = chunk_26[0]
        v_78 = chunk_26[1]
        chunk_26 = None
        v_79 = torch.nn.functional.layer_norm(
            v_78,
            (768,),
            l_self_modules_blocks_modules_26_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_26_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_,
            1e-05,
        )
        v_78 = l_self_modules_blocks_modules_26_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_26_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = (None)
        transpose_53 = v_79.transpose(-1, -2)
        v_79 = None
        v_80 = torch._C._nn.linear(
            transpose_53,
            l_self_modules_blocks_modules_26_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_26_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_,
        )
        transpose_53 = l_self_modules_blocks_modules_26_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_26_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = (None)
        transpose_54 = v_80.transpose(-1, -2)
        v_80 = None
        x_187 = u_26 * transpose_54
        u_26 = transpose_54 = None
        x_188 = torch._C._nn.linear(
            x_187,
            l_self_modules_blocks_modules_26_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_26_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_187 = l_self_modules_blocks_modules_26_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_26_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_189 = torch.nn.functional.dropout(x_188, 0.0, False, False)
        x_188 = None
        x_190 = x_183 + x_189
        x_183 = x_189 = None
        layer_norm_54 = torch.nn.functional.layer_norm(
            x_190,
            (256,),
            l_self_modules_blocks_modules_27_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_27_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_27_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_27_modules_norm_parameters_bias_
        ) = None
        x_191 = torch._C._nn.linear(
            layer_norm_54,
            l_self_modules_blocks_modules_27_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_27_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_54 = l_self_modules_blocks_modules_27_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_27_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_192 = torch._C._nn.gelu(x_191, approximate="none")
        x_191 = None
        x_193 = torch.nn.functional.dropout(x_192, 0.0, False, False)
        x_192 = None
        chunk_27 = x_193.chunk(2, dim=-1)
        x_193 = None
        u_27 = chunk_27[0]
        v_81 = chunk_27[1]
        chunk_27 = None
        v_82 = torch.nn.functional.layer_norm(
            v_81,
            (768,),
            l_self_modules_blocks_modules_27_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_27_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_,
            1e-05,
        )
        v_81 = l_self_modules_blocks_modules_27_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_27_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = (None)
        transpose_55 = v_82.transpose(-1, -2)
        v_82 = None
        v_83 = torch._C._nn.linear(
            transpose_55,
            l_self_modules_blocks_modules_27_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_27_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_,
        )
        transpose_55 = l_self_modules_blocks_modules_27_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_27_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = (None)
        transpose_56 = v_83.transpose(-1, -2)
        v_83 = None
        x_194 = u_27 * transpose_56
        u_27 = transpose_56 = None
        x_195 = torch._C._nn.linear(
            x_194,
            l_self_modules_blocks_modules_27_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_27_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_194 = l_self_modules_blocks_modules_27_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_27_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_196 = torch.nn.functional.dropout(x_195, 0.0, False, False)
        x_195 = None
        x_197 = x_190 + x_196
        x_190 = x_196 = None
        layer_norm_56 = torch.nn.functional.layer_norm(
            x_197,
            (256,),
            l_self_modules_blocks_modules_28_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_28_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_28_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_28_modules_norm_parameters_bias_
        ) = None
        x_198 = torch._C._nn.linear(
            layer_norm_56,
            l_self_modules_blocks_modules_28_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_28_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_56 = l_self_modules_blocks_modules_28_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_28_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_199 = torch._C._nn.gelu(x_198, approximate="none")
        x_198 = None
        x_200 = torch.nn.functional.dropout(x_199, 0.0, False, False)
        x_199 = None
        chunk_28 = x_200.chunk(2, dim=-1)
        x_200 = None
        u_28 = chunk_28[0]
        v_84 = chunk_28[1]
        chunk_28 = None
        v_85 = torch.nn.functional.layer_norm(
            v_84,
            (768,),
            l_self_modules_blocks_modules_28_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_28_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_,
            1e-05,
        )
        v_84 = l_self_modules_blocks_modules_28_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_28_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = (None)
        transpose_57 = v_85.transpose(-1, -2)
        v_85 = None
        v_86 = torch._C._nn.linear(
            transpose_57,
            l_self_modules_blocks_modules_28_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_28_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_,
        )
        transpose_57 = l_self_modules_blocks_modules_28_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_28_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = (None)
        transpose_58 = v_86.transpose(-1, -2)
        v_86 = None
        x_201 = u_28 * transpose_58
        u_28 = transpose_58 = None
        x_202 = torch._C._nn.linear(
            x_201,
            l_self_modules_blocks_modules_28_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_28_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_201 = l_self_modules_blocks_modules_28_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_28_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_203 = torch.nn.functional.dropout(x_202, 0.0, False, False)
        x_202 = None
        x_204 = x_197 + x_203
        x_197 = x_203 = None
        layer_norm_58 = torch.nn.functional.layer_norm(
            x_204,
            (256,),
            l_self_modules_blocks_modules_29_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_29_modules_norm_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_29_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_29_modules_norm_parameters_bias_
        ) = None
        x_205 = torch._C._nn.linear(
            layer_norm_58,
            l_self_modules_blocks_modules_29_modules_mlp_channels_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_29_modules_mlp_channels_modules_fc1_parameters_bias_,
        )
        layer_norm_58 = l_self_modules_blocks_modules_29_modules_mlp_channels_modules_fc1_parameters_weight_ = l_self_modules_blocks_modules_29_modules_mlp_channels_modules_fc1_parameters_bias_ = (None)
        x_206 = torch._C._nn.gelu(x_205, approximate="none")
        x_205 = None
        x_207 = torch.nn.functional.dropout(x_206, 0.0, False, False)
        x_206 = None
        chunk_29 = x_207.chunk(2, dim=-1)
        x_207 = None
        u_29 = chunk_29[0]
        v_87 = chunk_29[1]
        chunk_29 = None
        v_88 = torch.nn.functional.layer_norm(
            v_87,
            (768,),
            l_self_modules_blocks_modules_29_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_29_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_,
            1e-05,
        )
        v_87 = l_self_modules_blocks_modules_29_modules_mlp_channels_modules_gate_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_29_modules_mlp_channels_modules_gate_modules_norm_parameters_bias_ = (None)
        transpose_59 = v_88.transpose(-1, -2)
        v_88 = None
        v_89 = torch._C._nn.linear(
            transpose_59,
            l_self_modules_blocks_modules_29_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_29_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_,
        )
        transpose_59 = l_self_modules_blocks_modules_29_modules_mlp_channels_modules_gate_modules_proj_parameters_weight_ = l_self_modules_blocks_modules_29_modules_mlp_channels_modules_gate_modules_proj_parameters_bias_ = (None)
        transpose_60 = v_89.transpose(-1, -2)
        v_89 = None
        x_208 = u_29 * transpose_60
        u_29 = transpose_60 = None
        x_209 = torch._C._nn.linear(
            x_208,
            l_self_modules_blocks_modules_29_modules_mlp_channels_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_29_modules_mlp_channels_modules_fc2_parameters_bias_,
        )
        x_208 = l_self_modules_blocks_modules_29_modules_mlp_channels_modules_fc2_parameters_weight_ = l_self_modules_blocks_modules_29_modules_mlp_channels_modules_fc2_parameters_bias_ = (None)
        x_210 = torch.nn.functional.dropout(x_209, 0.0, False, False)
        x_209 = None
        x_211 = x_204 + x_210
        x_204 = x_210 = None
        x_212 = torch.nn.functional.layer_norm(
            x_211,
            (256,),
            l_self_modules_norm_parameters_weight_,
            l_self_modules_norm_parameters_bias_,
            1e-06,
        )
        x_211 = (
            l_self_modules_norm_parameters_weight_
        ) = l_self_modules_norm_parameters_bias_ = None
        x_213 = x_212.mean(dim=1)
        x_212 = None
        x_214 = torch.nn.functional.dropout(x_213, 0.0, False, False)
        x_213 = None
        x_215 = torch._C._nn.linear(
            x_214,
            l_self_modules_head_parameters_weight_,
            l_self_modules_head_parameters_bias_,
        )
        x_214 = (
            l_self_modules_head_parameters_weight_
        ) = l_self_modules_head_parameters_bias_ = None
        return (x_215,)
