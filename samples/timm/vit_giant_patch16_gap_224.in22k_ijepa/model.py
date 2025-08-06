import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_x_: torch.Tensor,
        L_self_modules_patch_embed_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_patch_embed_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_parameters_pos_embed_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_24_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_24_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_24_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_24_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_24_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_24_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_24_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_24_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_25_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_25_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_25_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_25_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_25_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_25_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_25_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_25_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_26_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_26_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_26_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_26_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_26_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_26_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_26_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_26_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_27_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_27_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_27_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_27_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_27_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_27_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_27_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_27_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_27_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_27_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_27_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_27_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_28_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_28_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_28_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_28_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_28_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_28_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_28_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_28_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_28_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_28_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_28_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_28_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_29_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_29_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_29_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_29_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_29_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_29_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_29_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_29_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_29_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_29_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_29_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_29_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_30_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_30_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_30_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_30_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_30_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_30_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_30_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_30_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_30_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_30_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_30_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_30_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_31_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_31_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_31_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_31_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_31_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_31_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_31_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_31_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_31_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_31_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_31_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_31_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_32_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_32_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_32_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_32_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_32_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_32_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_32_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_32_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_32_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_32_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_32_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_32_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_33_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_33_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_33_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_33_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_33_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_33_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_33_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_33_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_33_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_33_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_33_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_33_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_34_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_34_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_34_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_34_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_34_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_34_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_34_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_34_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_34_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_34_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_34_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_34_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_35_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_35_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_35_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_35_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_35_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_35_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_35_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_35_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_35_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_35_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_35_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_35_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_36_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_36_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_36_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_36_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_36_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_36_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_36_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_36_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_36_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_36_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_36_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_36_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_37_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_37_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_37_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_37_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_37_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_37_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_37_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_37_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_37_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_37_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_37_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_37_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_38_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_38_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_38_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_38_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_38_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_38_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_38_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_38_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_38_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_38_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_38_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_38_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_39_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_39_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_39_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_39_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_39_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_39_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_39_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_39_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_39_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_39_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_39_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_39_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_num_prefix_tokens: torch.SymInt,
    ):
        l_x_ = L_x_
        l_self_modules_patch_embed_modules_proj_parameters_weight_ = (
            L_self_modules_patch_embed_modules_proj_parameters_weight_
        )
        l_self_modules_patch_embed_modules_proj_parameters_bias_ = (
            L_self_modules_patch_embed_modules_proj_parameters_bias_
        )
        l_self_parameters_pos_embed_ = L_self_parameters_pos_embed_
        l_self_modules_blocks_modules_0_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_0_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_ = (
            L_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_
        )
        l_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_0_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_0_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_6_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_6_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_bias_ = (
            L_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_bias_
        )
        l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_6_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_6_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_7_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_7_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_7_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_7_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_bias_ = (
            L_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_bias_
        )
        l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_ = (
            L_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_
        )
        l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_7_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_7_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_7_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_7_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_8_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_8_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_8_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_8_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_bias_ = (
            L_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_bias_
        )
        l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_ = (
            L_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_
        )
        l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_8_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_8_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_8_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_8_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_9_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_9_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_9_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_9_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_bias_ = (
            L_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_bias_
        )
        l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_ = (
            L_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_
        )
        l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_9_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_9_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_9_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_9_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_10_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_10_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_10_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_10_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_bias_ = (
            L_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_bias_
        )
        l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_10_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_10_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_10_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_10_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_11_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_11_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_11_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_11_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_bias_ = (
            L_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_bias_
        )
        l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_11_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_11_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_11_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_11_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_12_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_12_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_12_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_12_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_bias_ = (
            L_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_bias_
        )
        l_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_12_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_12_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_12_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_12_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_13_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_13_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_13_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_13_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_bias_ = (
            L_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_bias_
        )
        l_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_13_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_13_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_13_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_13_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_14_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_14_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_14_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_14_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_bias_ = (
            L_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_bias_
        )
        l_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_14_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_14_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_14_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_14_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_15_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_15_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_15_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_15_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_bias_ = (
            L_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_bias_
        )
        l_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_15_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_15_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_15_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_15_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_16_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_16_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_16_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_16_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_bias_ = (
            L_self_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_bias_
        )
        l_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_16_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_16_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_16_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_16_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_17_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_17_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_17_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_17_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_bias_ = (
            L_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_bias_
        )
        l_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_17_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_17_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_17_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_17_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_18_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_18_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_18_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_18_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_bias_ = (
            L_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_bias_
        )
        l_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_18_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_18_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_18_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_18_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_19_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_19_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_19_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_19_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_bias_ = (
            L_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_bias_
        )
        l_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_19_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_19_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_19_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_19_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_20_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_20_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_20_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_20_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_bias_ = (
            L_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_bias_
        )
        l_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_20_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_20_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_20_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_20_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_21_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_21_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_21_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_21_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_bias_ = (
            L_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_bias_
        )
        l_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_21_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_21_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_21_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_21_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_22_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_22_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_22_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_22_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_bias_ = (
            L_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_bias_
        )
        l_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_22_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_22_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_22_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_22_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_23_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_23_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_23_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_23_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_bias_ = (
            L_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_bias_
        )
        l_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_23_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_23_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_23_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_23_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_24_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_24_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_24_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_24_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_24_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_24_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_24_modules_attn_modules_qkv_parameters_bias_ = (
            L_self_modules_blocks_modules_24_modules_attn_modules_qkv_parameters_bias_
        )
        l_self_modules_blocks_modules_24_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_24_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_24_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_24_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_24_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_24_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_24_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_24_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_25_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_25_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_25_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_25_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_25_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_25_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_25_modules_attn_modules_qkv_parameters_bias_ = (
            L_self_modules_blocks_modules_25_modules_attn_modules_qkv_parameters_bias_
        )
        l_self_modules_blocks_modules_25_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_25_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_25_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_25_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_25_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_25_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_25_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_25_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_26_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_26_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_26_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_26_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_26_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_26_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_26_modules_attn_modules_qkv_parameters_bias_ = (
            L_self_modules_blocks_modules_26_modules_attn_modules_qkv_parameters_bias_
        )
        l_self_modules_blocks_modules_26_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_26_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_26_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_26_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_26_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_26_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_26_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_26_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_27_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_27_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_27_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_27_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_27_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_27_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_27_modules_attn_modules_qkv_parameters_bias_ = (
            L_self_modules_blocks_modules_27_modules_attn_modules_qkv_parameters_bias_
        )
        l_self_modules_blocks_modules_27_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_27_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_27_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_27_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_27_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_27_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_27_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_27_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_27_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_27_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_27_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_27_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_27_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_27_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_27_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_27_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_28_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_28_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_28_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_28_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_28_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_28_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_28_modules_attn_modules_qkv_parameters_bias_ = (
            L_self_modules_blocks_modules_28_modules_attn_modules_qkv_parameters_bias_
        )
        l_self_modules_blocks_modules_28_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_28_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_28_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_28_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_28_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_28_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_28_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_28_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_28_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_28_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_28_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_28_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_28_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_28_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_28_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_28_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_29_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_29_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_29_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_29_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_29_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_29_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_29_modules_attn_modules_qkv_parameters_bias_ = (
            L_self_modules_blocks_modules_29_modules_attn_modules_qkv_parameters_bias_
        )
        l_self_modules_blocks_modules_29_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_29_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_29_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_29_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_29_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_29_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_29_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_29_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_29_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_29_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_29_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_29_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_29_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_29_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_29_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_29_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_30_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_30_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_30_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_30_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_30_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_30_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_30_modules_attn_modules_qkv_parameters_bias_ = (
            L_self_modules_blocks_modules_30_modules_attn_modules_qkv_parameters_bias_
        )
        l_self_modules_blocks_modules_30_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_30_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_30_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_30_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_30_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_30_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_30_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_30_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_30_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_30_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_30_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_30_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_30_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_30_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_30_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_30_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_31_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_31_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_31_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_31_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_31_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_31_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_31_modules_attn_modules_qkv_parameters_bias_ = (
            L_self_modules_blocks_modules_31_modules_attn_modules_qkv_parameters_bias_
        )
        l_self_modules_blocks_modules_31_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_31_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_31_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_31_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_31_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_31_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_31_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_31_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_31_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_31_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_31_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_31_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_31_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_31_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_31_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_31_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_32_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_32_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_32_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_32_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_32_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_32_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_32_modules_attn_modules_qkv_parameters_bias_ = (
            L_self_modules_blocks_modules_32_modules_attn_modules_qkv_parameters_bias_
        )
        l_self_modules_blocks_modules_32_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_32_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_32_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_32_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_32_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_32_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_32_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_32_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_32_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_32_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_32_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_32_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_32_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_32_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_32_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_32_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_33_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_33_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_33_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_33_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_33_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_33_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_33_modules_attn_modules_qkv_parameters_bias_ = (
            L_self_modules_blocks_modules_33_modules_attn_modules_qkv_parameters_bias_
        )
        l_self_modules_blocks_modules_33_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_33_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_33_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_33_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_33_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_33_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_33_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_33_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_33_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_33_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_33_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_33_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_33_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_33_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_33_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_33_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_34_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_34_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_34_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_34_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_34_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_34_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_34_modules_attn_modules_qkv_parameters_bias_ = (
            L_self_modules_blocks_modules_34_modules_attn_modules_qkv_parameters_bias_
        )
        l_self_modules_blocks_modules_34_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_34_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_34_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_34_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_34_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_34_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_34_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_34_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_34_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_34_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_34_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_34_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_34_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_34_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_34_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_34_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_35_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_35_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_35_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_35_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_35_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_35_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_35_modules_attn_modules_qkv_parameters_bias_ = (
            L_self_modules_blocks_modules_35_modules_attn_modules_qkv_parameters_bias_
        )
        l_self_modules_blocks_modules_35_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_35_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_35_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_35_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_35_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_35_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_35_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_35_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_35_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_35_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_35_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_35_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_35_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_35_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_35_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_35_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_36_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_36_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_36_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_36_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_36_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_36_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_36_modules_attn_modules_qkv_parameters_bias_ = (
            L_self_modules_blocks_modules_36_modules_attn_modules_qkv_parameters_bias_
        )
        l_self_modules_blocks_modules_36_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_36_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_36_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_36_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_36_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_36_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_36_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_36_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_36_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_36_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_36_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_36_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_36_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_36_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_36_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_36_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_37_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_37_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_37_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_37_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_37_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_37_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_37_modules_attn_modules_qkv_parameters_bias_ = (
            L_self_modules_blocks_modules_37_modules_attn_modules_qkv_parameters_bias_
        )
        l_self_modules_blocks_modules_37_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_37_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_37_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_37_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_37_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_37_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_37_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_37_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_37_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_37_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_37_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_37_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_37_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_37_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_37_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_37_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_38_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_38_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_38_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_38_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_38_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_38_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_38_modules_attn_modules_qkv_parameters_bias_ = (
            L_self_modules_blocks_modules_38_modules_attn_modules_qkv_parameters_bias_
        )
        l_self_modules_blocks_modules_38_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_38_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_38_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_38_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_38_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_38_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_38_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_38_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_38_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_38_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_38_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_38_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_38_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_38_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_38_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_38_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_39_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_39_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_39_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_39_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_39_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_39_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_39_modules_attn_modules_qkv_parameters_bias_ = (
            L_self_modules_blocks_modules_39_modules_attn_modules_qkv_parameters_bias_
        )
        l_self_modules_blocks_modules_39_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_39_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_39_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_39_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_39_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_39_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_39_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_39_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_39_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_39_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_39_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_39_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_39_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_39_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_39_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_39_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_norm_parameters_weight_ = L_self_modules_norm_parameters_weight_
        l_self_modules_norm_parameters_bias_ = L_self_modules_norm_parameters_bias_
        l_self_num_prefix_tokens = L_self_num_prefix_tokens
        x = torch.conv2d(
            l_x_,
            l_self_modules_patch_embed_modules_proj_parameters_weight_,
            l_self_modules_patch_embed_modules_proj_parameters_bias_,
            (16, 16),
            (0, 0),
            (1, 1),
            1,
        )
        l_x_ = (
            l_self_modules_patch_embed_modules_proj_parameters_weight_
        ) = l_self_modules_patch_embed_modules_proj_parameters_bias_ = None
        flatten = x.flatten(2)
        x = None
        x_1 = flatten.transpose(1, 2)
        flatten = None
        x_2 = x_1 + l_self_parameters_pos_embed_
        x_1 = l_self_parameters_pos_embed_ = None
        x_3 = torch.nn.functional.dropout(x_2, 0.0, False, False)
        x_2 = None
        x_4 = torch.nn.functional.layer_norm(
            x_3,
            (1408,),
            l_self_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_0_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_0_modules_norm1_parameters_bias_
        ) = None
        linear = torch._C._nn.linear(
            x_4,
            l_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_,
        )
        x_4 = (
            l_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape = linear.reshape(1, 196, 3, 16, 88)
        linear = None
        qkv = reshape.permute(2, 0, 3, 1, 4)
        reshape = None
        unbind = qkv.unbind(0)
        qkv = None
        q = unbind[0]
        k = unbind[1]
        v = unbind[2]
        unbind = None
        x_5 = torch._C._nn.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0
        )
        q = k = v = None
        transpose_1 = x_5.transpose(1, 2)
        x_5 = None
        x_6 = transpose_1.reshape(1, 196, 1408)
        transpose_1 = None
        x_7 = torch._C._nn.linear(
            x_6,
            l_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_6 = (
            l_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_8 = torch.nn.functional.dropout(x_7, 0.0, False, False)
        x_7 = None
        x_9 = x_3 + x_8
        x_3 = x_8 = None
        x_10 = torch.nn.functional.layer_norm(
            x_9,
            (1408,),
            l_self_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_0_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_0_modules_norm2_parameters_bias_
        ) = None
        x_11 = torch._C._nn.linear(
            x_10,
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_10 = (
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_12 = torch._C._nn.gelu(x_11, approximate="none")
        x_11 = None
        x_13 = torch.nn.functional.dropout(x_12, 0.0, False, False)
        x_12 = None
        x_14 = torch._C._nn.linear(
            x_13,
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_13 = (
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_15 = torch.nn.functional.dropout(x_14, 0.0, False, False)
        x_14 = None
        x_16 = x_9 + x_15
        x_9 = x_15 = None
        x_17 = torch.nn.functional.layer_norm(
            x_16,
            (1408,),
            l_self_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_1_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_1_modules_norm1_parameters_bias_
        ) = None
        linear_4 = torch._C._nn.linear(
            x_17,
            l_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_,
        )
        x_17 = (
            l_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_2 = linear_4.reshape(1, 196, 3, 16, 88)
        linear_4 = None
        qkv_1 = reshape_2.permute(2, 0, 3, 1, 4)
        reshape_2 = None
        unbind_1 = qkv_1.unbind(0)
        qkv_1 = None
        q_1 = unbind_1[0]
        k_1 = unbind_1[1]
        v_1 = unbind_1[2]
        unbind_1 = None
        x_18 = torch._C._nn.scaled_dot_product_attention(
            q_1, k_1, v_1, attn_mask=None, dropout_p=0.0
        )
        q_1 = k_1 = v_1 = None
        transpose_2 = x_18.transpose(1, 2)
        x_18 = None
        x_19 = transpose_2.reshape(1, 196, 1408)
        transpose_2 = None
        x_20 = torch._C._nn.linear(
            x_19,
            l_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_19 = (
            l_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_21 = torch.nn.functional.dropout(x_20, 0.0, False, False)
        x_20 = None
        x_22 = x_16 + x_21
        x_16 = x_21 = None
        x_23 = torch.nn.functional.layer_norm(
            x_22,
            (1408,),
            l_self_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_1_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_1_modules_norm2_parameters_bias_
        ) = None
        x_24 = torch._C._nn.linear(
            x_23,
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_23 = (
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_25 = torch._C._nn.gelu(x_24, approximate="none")
        x_24 = None
        x_26 = torch.nn.functional.dropout(x_25, 0.0, False, False)
        x_25 = None
        x_27 = torch._C._nn.linear(
            x_26,
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_26 = (
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_28 = torch.nn.functional.dropout(x_27, 0.0, False, False)
        x_27 = None
        x_29 = x_22 + x_28
        x_22 = x_28 = None
        x_30 = torch.nn.functional.layer_norm(
            x_29,
            (1408,),
            l_self_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_2_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_2_modules_norm1_parameters_bias_
        ) = None
        linear_8 = torch._C._nn.linear(
            x_30,
            l_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_,
        )
        x_30 = (
            l_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_4 = linear_8.reshape(1, 196, 3, 16, 88)
        linear_8 = None
        qkv_2 = reshape_4.permute(2, 0, 3, 1, 4)
        reshape_4 = None
        unbind_2 = qkv_2.unbind(0)
        qkv_2 = None
        q_2 = unbind_2[0]
        k_2 = unbind_2[1]
        v_2 = unbind_2[2]
        unbind_2 = None
        x_31 = torch._C._nn.scaled_dot_product_attention(
            q_2, k_2, v_2, attn_mask=None, dropout_p=0.0
        )
        q_2 = k_2 = v_2 = None
        transpose_3 = x_31.transpose(1, 2)
        x_31 = None
        x_32 = transpose_3.reshape(1, 196, 1408)
        transpose_3 = None
        x_33 = torch._C._nn.linear(
            x_32,
            l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_,
        )
        x_32 = (
            l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_34 = torch.nn.functional.dropout(x_33, 0.0, False, False)
        x_33 = None
        x_35 = x_29 + x_34
        x_29 = x_34 = None
        x_36 = torch.nn.functional.layer_norm(
            x_35,
            (1408,),
            l_self_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_2_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_2_modules_norm2_parameters_bias_
        ) = None
        x_37 = torch._C._nn.linear(
            x_36,
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_36 = (
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_38 = torch._C._nn.gelu(x_37, approximate="none")
        x_37 = None
        x_39 = torch.nn.functional.dropout(x_38, 0.0, False, False)
        x_38 = None
        x_40 = torch._C._nn.linear(
            x_39,
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_39 = (
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_41 = torch.nn.functional.dropout(x_40, 0.0, False, False)
        x_40 = None
        x_42 = x_35 + x_41
        x_35 = x_41 = None
        x_43 = torch.nn.functional.layer_norm(
            x_42,
            (1408,),
            l_self_modules_blocks_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_3_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_3_modules_norm1_parameters_bias_
        ) = None
        linear_12 = torch._C._nn.linear(
            x_43,
            l_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_bias_,
        )
        x_43 = (
            l_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_6 = linear_12.reshape(1, 196, 3, 16, 88)
        linear_12 = None
        qkv_3 = reshape_6.permute(2, 0, 3, 1, 4)
        reshape_6 = None
        unbind_3 = qkv_3.unbind(0)
        qkv_3 = None
        q_3 = unbind_3[0]
        k_3 = unbind_3[1]
        v_3 = unbind_3[2]
        unbind_3 = None
        x_44 = torch._C._nn.scaled_dot_product_attention(
            q_3, k_3, v_3, attn_mask=None, dropout_p=0.0
        )
        q_3 = k_3 = v_3 = None
        transpose_4 = x_44.transpose(1, 2)
        x_44 = None
        x_45 = transpose_4.reshape(1, 196, 1408)
        transpose_4 = None
        x_46 = torch._C._nn.linear(
            x_45,
            l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_,
        )
        x_45 = (
            l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_47 = torch.nn.functional.dropout(x_46, 0.0, False, False)
        x_46 = None
        x_48 = x_42 + x_47
        x_42 = x_47 = None
        x_49 = torch.nn.functional.layer_norm(
            x_48,
            (1408,),
            l_self_modules_blocks_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_3_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_3_modules_norm2_parameters_bias_
        ) = None
        x_50 = torch._C._nn.linear(
            x_49,
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_49 = (
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_51 = torch._C._nn.gelu(x_50, approximate="none")
        x_50 = None
        x_52 = torch.nn.functional.dropout(x_51, 0.0, False, False)
        x_51 = None
        x_53 = torch._C._nn.linear(
            x_52,
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_52 = (
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_54 = torch.nn.functional.dropout(x_53, 0.0, False, False)
        x_53 = None
        x_55 = x_48 + x_54
        x_48 = x_54 = None
        x_56 = torch.nn.functional.layer_norm(
            x_55,
            (1408,),
            l_self_modules_blocks_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_4_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_4_modules_norm1_parameters_bias_
        ) = None
        linear_16 = torch._C._nn.linear(
            x_56,
            l_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_bias_,
        )
        x_56 = (
            l_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_8 = linear_16.reshape(1, 196, 3, 16, 88)
        linear_16 = None
        qkv_4 = reshape_8.permute(2, 0, 3, 1, 4)
        reshape_8 = None
        unbind_4 = qkv_4.unbind(0)
        qkv_4 = None
        q_4 = unbind_4[0]
        k_4 = unbind_4[1]
        v_4 = unbind_4[2]
        unbind_4 = None
        x_57 = torch._C._nn.scaled_dot_product_attention(
            q_4, k_4, v_4, attn_mask=None, dropout_p=0.0
        )
        q_4 = k_4 = v_4 = None
        transpose_5 = x_57.transpose(1, 2)
        x_57 = None
        x_58 = transpose_5.reshape(1, 196, 1408)
        transpose_5 = None
        x_59 = torch._C._nn.linear(
            x_58,
            l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_,
        )
        x_58 = (
            l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_60 = torch.nn.functional.dropout(x_59, 0.0, False, False)
        x_59 = None
        x_61 = x_55 + x_60
        x_55 = x_60 = None
        x_62 = torch.nn.functional.layer_norm(
            x_61,
            (1408,),
            l_self_modules_blocks_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_4_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_4_modules_norm2_parameters_bias_
        ) = None
        x_63 = torch._C._nn.linear(
            x_62,
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_62 = (
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_64 = torch._C._nn.gelu(x_63, approximate="none")
        x_63 = None
        x_65 = torch.nn.functional.dropout(x_64, 0.0, False, False)
        x_64 = None
        x_66 = torch._C._nn.linear(
            x_65,
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_65 = (
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_67 = torch.nn.functional.dropout(x_66, 0.0, False, False)
        x_66 = None
        x_68 = x_61 + x_67
        x_61 = x_67 = None
        x_69 = torch.nn.functional.layer_norm(
            x_68,
            (1408,),
            l_self_modules_blocks_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_5_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_5_modules_norm1_parameters_bias_
        ) = None
        linear_20 = torch._C._nn.linear(
            x_69,
            l_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_bias_,
        )
        x_69 = (
            l_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_10 = linear_20.reshape(1, 196, 3, 16, 88)
        linear_20 = None
        qkv_5 = reshape_10.permute(2, 0, 3, 1, 4)
        reshape_10 = None
        unbind_5 = qkv_5.unbind(0)
        qkv_5 = None
        q_5 = unbind_5[0]
        k_5 = unbind_5[1]
        v_5 = unbind_5[2]
        unbind_5 = None
        x_70 = torch._C._nn.scaled_dot_product_attention(
            q_5, k_5, v_5, attn_mask=None, dropout_p=0.0
        )
        q_5 = k_5 = v_5 = None
        transpose_6 = x_70.transpose(1, 2)
        x_70 = None
        x_71 = transpose_6.reshape(1, 196, 1408)
        transpose_6 = None
        x_72 = torch._C._nn.linear(
            x_71,
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_,
        )
        x_71 = (
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_73 = torch.nn.functional.dropout(x_72, 0.0, False, False)
        x_72 = None
        x_74 = x_68 + x_73
        x_68 = x_73 = None
        x_75 = torch.nn.functional.layer_norm(
            x_74,
            (1408,),
            l_self_modules_blocks_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_5_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_5_modules_norm2_parameters_bias_
        ) = None
        x_76 = torch._C._nn.linear(
            x_75,
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_75 = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_77 = torch._C._nn.gelu(x_76, approximate="none")
        x_76 = None
        x_78 = torch.nn.functional.dropout(x_77, 0.0, False, False)
        x_77 = None
        x_79 = torch._C._nn.linear(
            x_78,
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_78 = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_80 = torch.nn.functional.dropout(x_79, 0.0, False, False)
        x_79 = None
        x_81 = x_74 + x_80
        x_74 = x_80 = None
        x_82 = torch.nn.functional.layer_norm(
            x_81,
            (1408,),
            l_self_modules_blocks_modules_6_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_6_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_6_modules_norm1_parameters_bias_
        ) = None
        linear_24 = torch._C._nn.linear(
            x_82,
            l_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_bias_,
        )
        x_82 = (
            l_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_12 = linear_24.reshape(1, 196, 3, 16, 88)
        linear_24 = None
        qkv_6 = reshape_12.permute(2, 0, 3, 1, 4)
        reshape_12 = None
        unbind_6 = qkv_6.unbind(0)
        qkv_6 = None
        q_6 = unbind_6[0]
        k_6 = unbind_6[1]
        v_6 = unbind_6[2]
        unbind_6 = None
        x_83 = torch._C._nn.scaled_dot_product_attention(
            q_6, k_6, v_6, attn_mask=None, dropout_p=0.0
        )
        q_6 = k_6 = v_6 = None
        transpose_7 = x_83.transpose(1, 2)
        x_83 = None
        x_84 = transpose_7.reshape(1, 196, 1408)
        transpose_7 = None
        x_85 = torch._C._nn.linear(
            x_84,
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_,
        )
        x_84 = (
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_86 = torch.nn.functional.dropout(x_85, 0.0, False, False)
        x_85 = None
        x_87 = x_81 + x_86
        x_81 = x_86 = None
        x_88 = torch.nn.functional.layer_norm(
            x_87,
            (1408,),
            l_self_modules_blocks_modules_6_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_6_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_6_modules_norm2_parameters_bias_
        ) = None
        x_89 = torch._C._nn.linear(
            x_88,
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_88 = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_90 = torch._C._nn.gelu(x_89, approximate="none")
        x_89 = None
        x_91 = torch.nn.functional.dropout(x_90, 0.0, False, False)
        x_90 = None
        x_92 = torch._C._nn.linear(
            x_91,
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_91 = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_93 = torch.nn.functional.dropout(x_92, 0.0, False, False)
        x_92 = None
        x_94 = x_87 + x_93
        x_87 = x_93 = None
        x_95 = torch.nn.functional.layer_norm(
            x_94,
            (1408,),
            l_self_modules_blocks_modules_7_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_7_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_7_modules_norm1_parameters_bias_
        ) = None
        linear_28 = torch._C._nn.linear(
            x_95,
            l_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_bias_,
        )
        x_95 = (
            l_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_14 = linear_28.reshape(1, 196, 3, 16, 88)
        linear_28 = None
        qkv_7 = reshape_14.permute(2, 0, 3, 1, 4)
        reshape_14 = None
        unbind_7 = qkv_7.unbind(0)
        qkv_7 = None
        q_7 = unbind_7[0]
        k_7 = unbind_7[1]
        v_7 = unbind_7[2]
        unbind_7 = None
        x_96 = torch._C._nn.scaled_dot_product_attention(
            q_7, k_7, v_7, attn_mask=None, dropout_p=0.0
        )
        q_7 = k_7 = v_7 = None
        transpose_8 = x_96.transpose(1, 2)
        x_96 = None
        x_97 = transpose_8.reshape(1, 196, 1408)
        transpose_8 = None
        x_98 = torch._C._nn.linear(
            x_97,
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_,
        )
        x_97 = (
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_99 = torch.nn.functional.dropout(x_98, 0.0, False, False)
        x_98 = None
        x_100 = x_94 + x_99
        x_94 = x_99 = None
        x_101 = torch.nn.functional.layer_norm(
            x_100,
            (1408,),
            l_self_modules_blocks_modules_7_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_7_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_7_modules_norm2_parameters_bias_
        ) = None
        x_102 = torch._C._nn.linear(
            x_101,
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_101 = (
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_103 = torch._C._nn.gelu(x_102, approximate="none")
        x_102 = None
        x_104 = torch.nn.functional.dropout(x_103, 0.0, False, False)
        x_103 = None
        x_105 = torch._C._nn.linear(
            x_104,
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_104 = (
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_106 = torch.nn.functional.dropout(x_105, 0.0, False, False)
        x_105 = None
        x_107 = x_100 + x_106
        x_100 = x_106 = None
        x_108 = torch.nn.functional.layer_norm(
            x_107,
            (1408,),
            l_self_modules_blocks_modules_8_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_8_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_8_modules_norm1_parameters_bias_
        ) = None
        linear_32 = torch._C._nn.linear(
            x_108,
            l_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_bias_,
        )
        x_108 = (
            l_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_16 = linear_32.reshape(1, 196, 3, 16, 88)
        linear_32 = None
        qkv_8 = reshape_16.permute(2, 0, 3, 1, 4)
        reshape_16 = None
        unbind_8 = qkv_8.unbind(0)
        qkv_8 = None
        q_8 = unbind_8[0]
        k_8 = unbind_8[1]
        v_8 = unbind_8[2]
        unbind_8 = None
        x_109 = torch._C._nn.scaled_dot_product_attention(
            q_8, k_8, v_8, attn_mask=None, dropout_p=0.0
        )
        q_8 = k_8 = v_8 = None
        transpose_9 = x_109.transpose(1, 2)
        x_109 = None
        x_110 = transpose_9.reshape(1, 196, 1408)
        transpose_9 = None
        x_111 = torch._C._nn.linear(
            x_110,
            l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_,
        )
        x_110 = (
            l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_112 = torch.nn.functional.dropout(x_111, 0.0, False, False)
        x_111 = None
        x_113 = x_107 + x_112
        x_107 = x_112 = None
        x_114 = torch.nn.functional.layer_norm(
            x_113,
            (1408,),
            l_self_modules_blocks_modules_8_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_8_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_8_modules_norm2_parameters_bias_
        ) = None
        x_115 = torch._C._nn.linear(
            x_114,
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_114 = (
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_116 = torch._C._nn.gelu(x_115, approximate="none")
        x_115 = None
        x_117 = torch.nn.functional.dropout(x_116, 0.0, False, False)
        x_116 = None
        x_118 = torch._C._nn.linear(
            x_117,
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_117 = (
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_119 = torch.nn.functional.dropout(x_118, 0.0, False, False)
        x_118 = None
        x_120 = x_113 + x_119
        x_113 = x_119 = None
        x_121 = torch.nn.functional.layer_norm(
            x_120,
            (1408,),
            l_self_modules_blocks_modules_9_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_9_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_9_modules_norm1_parameters_bias_
        ) = None
        linear_36 = torch._C._nn.linear(
            x_121,
            l_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_bias_,
        )
        x_121 = (
            l_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_18 = linear_36.reshape(1, 196, 3, 16, 88)
        linear_36 = None
        qkv_9 = reshape_18.permute(2, 0, 3, 1, 4)
        reshape_18 = None
        unbind_9 = qkv_9.unbind(0)
        qkv_9 = None
        q_9 = unbind_9[0]
        k_9 = unbind_9[1]
        v_9 = unbind_9[2]
        unbind_9 = None
        x_122 = torch._C._nn.scaled_dot_product_attention(
            q_9, k_9, v_9, attn_mask=None, dropout_p=0.0
        )
        q_9 = k_9 = v_9 = None
        transpose_10 = x_122.transpose(1, 2)
        x_122 = None
        x_123 = transpose_10.reshape(1, 196, 1408)
        transpose_10 = None
        x_124 = torch._C._nn.linear(
            x_123,
            l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_,
        )
        x_123 = (
            l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_125 = torch.nn.functional.dropout(x_124, 0.0, False, False)
        x_124 = None
        x_126 = x_120 + x_125
        x_120 = x_125 = None
        x_127 = torch.nn.functional.layer_norm(
            x_126,
            (1408,),
            l_self_modules_blocks_modules_9_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_9_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_9_modules_norm2_parameters_bias_
        ) = None
        x_128 = torch._C._nn.linear(
            x_127,
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_127 = (
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_129 = torch._C._nn.gelu(x_128, approximate="none")
        x_128 = None
        x_130 = torch.nn.functional.dropout(x_129, 0.0, False, False)
        x_129 = None
        x_131 = torch._C._nn.linear(
            x_130,
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_130 = (
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_132 = torch.nn.functional.dropout(x_131, 0.0, False, False)
        x_131 = None
        x_133 = x_126 + x_132
        x_126 = x_132 = None
        x_134 = torch.nn.functional.layer_norm(
            x_133,
            (1408,),
            l_self_modules_blocks_modules_10_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_10_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_10_modules_norm1_parameters_bias_
        ) = None
        linear_40 = torch._C._nn.linear(
            x_134,
            l_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_bias_,
        )
        x_134 = (
            l_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_20 = linear_40.reshape(1, 196, 3, 16, 88)
        linear_40 = None
        qkv_10 = reshape_20.permute(2, 0, 3, 1, 4)
        reshape_20 = None
        unbind_10 = qkv_10.unbind(0)
        qkv_10 = None
        q_10 = unbind_10[0]
        k_10 = unbind_10[1]
        v_10 = unbind_10[2]
        unbind_10 = None
        x_135 = torch._C._nn.scaled_dot_product_attention(
            q_10, k_10, v_10, attn_mask=None, dropout_p=0.0
        )
        q_10 = k_10 = v_10 = None
        transpose_11 = x_135.transpose(1, 2)
        x_135 = None
        x_136 = transpose_11.reshape(1, 196, 1408)
        transpose_11 = None
        x_137 = torch._C._nn.linear(
            x_136,
            l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_,
        )
        x_136 = l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_138 = torch.nn.functional.dropout(x_137, 0.0, False, False)
        x_137 = None
        x_139 = x_133 + x_138
        x_133 = x_138 = None
        x_140 = torch.nn.functional.layer_norm(
            x_139,
            (1408,),
            l_self_modules_blocks_modules_10_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_10_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_10_modules_norm2_parameters_bias_
        ) = None
        x_141 = torch._C._nn.linear(
            x_140,
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_140 = (
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_142 = torch._C._nn.gelu(x_141, approximate="none")
        x_141 = None
        x_143 = torch.nn.functional.dropout(x_142, 0.0, False, False)
        x_142 = None
        x_144 = torch._C._nn.linear(
            x_143,
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_143 = (
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_145 = torch.nn.functional.dropout(x_144, 0.0, False, False)
        x_144 = None
        x_146 = x_139 + x_145
        x_139 = x_145 = None
        x_147 = torch.nn.functional.layer_norm(
            x_146,
            (1408,),
            l_self_modules_blocks_modules_11_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_11_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_11_modules_norm1_parameters_bias_
        ) = None
        linear_44 = torch._C._nn.linear(
            x_147,
            l_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_bias_,
        )
        x_147 = (
            l_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_22 = linear_44.reshape(1, 196, 3, 16, 88)
        linear_44 = None
        qkv_11 = reshape_22.permute(2, 0, 3, 1, 4)
        reshape_22 = None
        unbind_11 = qkv_11.unbind(0)
        qkv_11 = None
        q_11 = unbind_11[0]
        k_11 = unbind_11[1]
        v_11 = unbind_11[2]
        unbind_11 = None
        x_148 = torch._C._nn.scaled_dot_product_attention(
            q_11, k_11, v_11, attn_mask=None, dropout_p=0.0
        )
        q_11 = k_11 = v_11 = None
        transpose_12 = x_148.transpose(1, 2)
        x_148 = None
        x_149 = transpose_12.reshape(1, 196, 1408)
        transpose_12 = None
        x_150 = torch._C._nn.linear(
            x_149,
            l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_,
        )
        x_149 = l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_151 = torch.nn.functional.dropout(x_150, 0.0, False, False)
        x_150 = None
        x_152 = x_146 + x_151
        x_146 = x_151 = None
        x_153 = torch.nn.functional.layer_norm(
            x_152,
            (1408,),
            l_self_modules_blocks_modules_11_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_11_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_11_modules_norm2_parameters_bias_
        ) = None
        x_154 = torch._C._nn.linear(
            x_153,
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_153 = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_155 = torch._C._nn.gelu(x_154, approximate="none")
        x_154 = None
        x_156 = torch.nn.functional.dropout(x_155, 0.0, False, False)
        x_155 = None
        x_157 = torch._C._nn.linear(
            x_156,
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_156 = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_158 = torch.nn.functional.dropout(x_157, 0.0, False, False)
        x_157 = None
        x_159 = x_152 + x_158
        x_152 = x_158 = None
        x_160 = torch.nn.functional.layer_norm(
            x_159,
            (1408,),
            l_self_modules_blocks_modules_12_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_12_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_12_modules_norm1_parameters_bias_
        ) = None
        linear_48 = torch._C._nn.linear(
            x_160,
            l_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_bias_,
        )
        x_160 = (
            l_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_24 = linear_48.reshape(1, 196, 3, 16, 88)
        linear_48 = None
        qkv_12 = reshape_24.permute(2, 0, 3, 1, 4)
        reshape_24 = None
        unbind_12 = qkv_12.unbind(0)
        qkv_12 = None
        q_12 = unbind_12[0]
        k_12 = unbind_12[1]
        v_12 = unbind_12[2]
        unbind_12 = None
        x_161 = torch._C._nn.scaled_dot_product_attention(
            q_12, k_12, v_12, attn_mask=None, dropout_p=0.0
        )
        q_12 = k_12 = v_12 = None
        transpose_13 = x_161.transpose(1, 2)
        x_161 = None
        x_162 = transpose_13.reshape(1, 196, 1408)
        transpose_13 = None
        x_163 = torch._C._nn.linear(
            x_162,
            l_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_,
        )
        x_162 = l_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_164 = torch.nn.functional.dropout(x_163, 0.0, False, False)
        x_163 = None
        x_165 = x_159 + x_164
        x_159 = x_164 = None
        x_166 = torch.nn.functional.layer_norm(
            x_165,
            (1408,),
            l_self_modules_blocks_modules_12_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_12_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_12_modules_norm2_parameters_bias_
        ) = None
        x_167 = torch._C._nn.linear(
            x_166,
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_166 = (
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_168 = torch._C._nn.gelu(x_167, approximate="none")
        x_167 = None
        x_169 = torch.nn.functional.dropout(x_168, 0.0, False, False)
        x_168 = None
        x_170 = torch._C._nn.linear(
            x_169,
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_169 = (
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_171 = torch.nn.functional.dropout(x_170, 0.0, False, False)
        x_170 = None
        x_172 = x_165 + x_171
        x_165 = x_171 = None
        x_173 = torch.nn.functional.layer_norm(
            x_172,
            (1408,),
            l_self_modules_blocks_modules_13_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_13_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_13_modules_norm1_parameters_bias_
        ) = None
        linear_52 = torch._C._nn.linear(
            x_173,
            l_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_bias_,
        )
        x_173 = (
            l_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_26 = linear_52.reshape(1, 196, 3, 16, 88)
        linear_52 = None
        qkv_13 = reshape_26.permute(2, 0, 3, 1, 4)
        reshape_26 = None
        unbind_13 = qkv_13.unbind(0)
        qkv_13 = None
        q_13 = unbind_13[0]
        k_13 = unbind_13[1]
        v_13 = unbind_13[2]
        unbind_13 = None
        x_174 = torch._C._nn.scaled_dot_product_attention(
            q_13, k_13, v_13, attn_mask=None, dropout_p=0.0
        )
        q_13 = k_13 = v_13 = None
        transpose_14 = x_174.transpose(1, 2)
        x_174 = None
        x_175 = transpose_14.reshape(1, 196, 1408)
        transpose_14 = None
        x_176 = torch._C._nn.linear(
            x_175,
            l_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_,
        )
        x_175 = l_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_177 = torch.nn.functional.dropout(x_176, 0.0, False, False)
        x_176 = None
        x_178 = x_172 + x_177
        x_172 = x_177 = None
        x_179 = torch.nn.functional.layer_norm(
            x_178,
            (1408,),
            l_self_modules_blocks_modules_13_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_13_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_13_modules_norm2_parameters_bias_
        ) = None
        x_180 = torch._C._nn.linear(
            x_179,
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_179 = (
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_181 = torch._C._nn.gelu(x_180, approximate="none")
        x_180 = None
        x_182 = torch.nn.functional.dropout(x_181, 0.0, False, False)
        x_181 = None
        x_183 = torch._C._nn.linear(
            x_182,
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_182 = (
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_184 = torch.nn.functional.dropout(x_183, 0.0, False, False)
        x_183 = None
        x_185 = x_178 + x_184
        x_178 = x_184 = None
        x_186 = torch.nn.functional.layer_norm(
            x_185,
            (1408,),
            l_self_modules_blocks_modules_14_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_14_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_14_modules_norm1_parameters_bias_
        ) = None
        linear_56 = torch._C._nn.linear(
            x_186,
            l_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_bias_,
        )
        x_186 = (
            l_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_28 = linear_56.reshape(1, 196, 3, 16, 88)
        linear_56 = None
        qkv_14 = reshape_28.permute(2, 0, 3, 1, 4)
        reshape_28 = None
        unbind_14 = qkv_14.unbind(0)
        qkv_14 = None
        q_14 = unbind_14[0]
        k_14 = unbind_14[1]
        v_14 = unbind_14[2]
        unbind_14 = None
        x_187 = torch._C._nn.scaled_dot_product_attention(
            q_14, k_14, v_14, attn_mask=None, dropout_p=0.0
        )
        q_14 = k_14 = v_14 = None
        transpose_15 = x_187.transpose(1, 2)
        x_187 = None
        x_188 = transpose_15.reshape(1, 196, 1408)
        transpose_15 = None
        x_189 = torch._C._nn.linear(
            x_188,
            l_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_bias_,
        )
        x_188 = l_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_190 = torch.nn.functional.dropout(x_189, 0.0, False, False)
        x_189 = None
        x_191 = x_185 + x_190
        x_185 = x_190 = None
        x_192 = torch.nn.functional.layer_norm(
            x_191,
            (1408,),
            l_self_modules_blocks_modules_14_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_14_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_14_modules_norm2_parameters_bias_
        ) = None
        x_193 = torch._C._nn.linear(
            x_192,
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_192 = (
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_194 = torch._C._nn.gelu(x_193, approximate="none")
        x_193 = None
        x_195 = torch.nn.functional.dropout(x_194, 0.0, False, False)
        x_194 = None
        x_196 = torch._C._nn.linear(
            x_195,
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_195 = (
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_197 = torch.nn.functional.dropout(x_196, 0.0, False, False)
        x_196 = None
        x_198 = x_191 + x_197
        x_191 = x_197 = None
        x_199 = torch.nn.functional.layer_norm(
            x_198,
            (1408,),
            l_self_modules_blocks_modules_15_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_15_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_15_modules_norm1_parameters_bias_
        ) = None
        linear_60 = torch._C._nn.linear(
            x_199,
            l_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_bias_,
        )
        x_199 = (
            l_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_30 = linear_60.reshape(1, 196, 3, 16, 88)
        linear_60 = None
        qkv_15 = reshape_30.permute(2, 0, 3, 1, 4)
        reshape_30 = None
        unbind_15 = qkv_15.unbind(0)
        qkv_15 = None
        q_15 = unbind_15[0]
        k_15 = unbind_15[1]
        v_15 = unbind_15[2]
        unbind_15 = None
        x_200 = torch._C._nn.scaled_dot_product_attention(
            q_15, k_15, v_15, attn_mask=None, dropout_p=0.0
        )
        q_15 = k_15 = v_15 = None
        transpose_16 = x_200.transpose(1, 2)
        x_200 = None
        x_201 = transpose_16.reshape(1, 196, 1408)
        transpose_16 = None
        x_202 = torch._C._nn.linear(
            x_201,
            l_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_bias_,
        )
        x_201 = l_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_203 = torch.nn.functional.dropout(x_202, 0.0, False, False)
        x_202 = None
        x_204 = x_198 + x_203
        x_198 = x_203 = None
        x_205 = torch.nn.functional.layer_norm(
            x_204,
            (1408,),
            l_self_modules_blocks_modules_15_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_15_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_15_modules_norm2_parameters_bias_
        ) = None
        x_206 = torch._C._nn.linear(
            x_205,
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_205 = (
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_207 = torch._C._nn.gelu(x_206, approximate="none")
        x_206 = None
        x_208 = torch.nn.functional.dropout(x_207, 0.0, False, False)
        x_207 = None
        x_209 = torch._C._nn.linear(
            x_208,
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_208 = (
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_210 = torch.nn.functional.dropout(x_209, 0.0, False, False)
        x_209 = None
        x_211 = x_204 + x_210
        x_204 = x_210 = None
        x_212 = torch.nn.functional.layer_norm(
            x_211,
            (1408,),
            l_self_modules_blocks_modules_16_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_16_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_16_modules_norm1_parameters_bias_
        ) = None
        linear_64 = torch._C._nn.linear(
            x_212,
            l_self_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_bias_,
        )
        x_212 = (
            l_self_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_32 = linear_64.reshape(1, 196, 3, 16, 88)
        linear_64 = None
        qkv_16 = reshape_32.permute(2, 0, 3, 1, 4)
        reshape_32 = None
        unbind_16 = qkv_16.unbind(0)
        qkv_16 = None
        q_16 = unbind_16[0]
        k_16 = unbind_16[1]
        v_16 = unbind_16[2]
        unbind_16 = None
        x_213 = torch._C._nn.scaled_dot_product_attention(
            q_16, k_16, v_16, attn_mask=None, dropout_p=0.0
        )
        q_16 = k_16 = v_16 = None
        transpose_17 = x_213.transpose(1, 2)
        x_213 = None
        x_214 = transpose_17.reshape(1, 196, 1408)
        transpose_17 = None
        x_215 = torch._C._nn.linear(
            x_214,
            l_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_bias_,
        )
        x_214 = l_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_216 = torch.nn.functional.dropout(x_215, 0.0, False, False)
        x_215 = None
        x_217 = x_211 + x_216
        x_211 = x_216 = None
        x_218 = torch.nn.functional.layer_norm(
            x_217,
            (1408,),
            l_self_modules_blocks_modules_16_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_16_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_16_modules_norm2_parameters_bias_
        ) = None
        x_219 = torch._C._nn.linear(
            x_218,
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_218 = (
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_220 = torch._C._nn.gelu(x_219, approximate="none")
        x_219 = None
        x_221 = torch.nn.functional.dropout(x_220, 0.0, False, False)
        x_220 = None
        x_222 = torch._C._nn.linear(
            x_221,
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_221 = (
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_223 = torch.nn.functional.dropout(x_222, 0.0, False, False)
        x_222 = None
        x_224 = x_217 + x_223
        x_217 = x_223 = None
        x_225 = torch.nn.functional.layer_norm(
            x_224,
            (1408,),
            l_self_modules_blocks_modules_17_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_17_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_17_modules_norm1_parameters_bias_
        ) = None
        linear_68 = torch._C._nn.linear(
            x_225,
            l_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_bias_,
        )
        x_225 = (
            l_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_34 = linear_68.reshape(1, 196, 3, 16, 88)
        linear_68 = None
        qkv_17 = reshape_34.permute(2, 0, 3, 1, 4)
        reshape_34 = None
        unbind_17 = qkv_17.unbind(0)
        qkv_17 = None
        q_17 = unbind_17[0]
        k_17 = unbind_17[1]
        v_17 = unbind_17[2]
        unbind_17 = None
        x_226 = torch._C._nn.scaled_dot_product_attention(
            q_17, k_17, v_17, attn_mask=None, dropout_p=0.0
        )
        q_17 = k_17 = v_17 = None
        transpose_18 = x_226.transpose(1, 2)
        x_226 = None
        x_227 = transpose_18.reshape(1, 196, 1408)
        transpose_18 = None
        x_228 = torch._C._nn.linear(
            x_227,
            l_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_bias_,
        )
        x_227 = l_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_229 = torch.nn.functional.dropout(x_228, 0.0, False, False)
        x_228 = None
        x_230 = x_224 + x_229
        x_224 = x_229 = None
        x_231 = torch.nn.functional.layer_norm(
            x_230,
            (1408,),
            l_self_modules_blocks_modules_17_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_17_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_17_modules_norm2_parameters_bias_
        ) = None
        x_232 = torch._C._nn.linear(
            x_231,
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_231 = (
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_233 = torch._C._nn.gelu(x_232, approximate="none")
        x_232 = None
        x_234 = torch.nn.functional.dropout(x_233, 0.0, False, False)
        x_233 = None
        x_235 = torch._C._nn.linear(
            x_234,
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_234 = (
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_236 = torch.nn.functional.dropout(x_235, 0.0, False, False)
        x_235 = None
        x_237 = x_230 + x_236
        x_230 = x_236 = None
        x_238 = torch.nn.functional.layer_norm(
            x_237,
            (1408,),
            l_self_modules_blocks_modules_18_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_18_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_18_modules_norm1_parameters_bias_
        ) = None
        linear_72 = torch._C._nn.linear(
            x_238,
            l_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_bias_,
        )
        x_238 = (
            l_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_36 = linear_72.reshape(1, 196, 3, 16, 88)
        linear_72 = None
        qkv_18 = reshape_36.permute(2, 0, 3, 1, 4)
        reshape_36 = None
        unbind_18 = qkv_18.unbind(0)
        qkv_18 = None
        q_18 = unbind_18[0]
        k_18 = unbind_18[1]
        v_18 = unbind_18[2]
        unbind_18 = None
        x_239 = torch._C._nn.scaled_dot_product_attention(
            q_18, k_18, v_18, attn_mask=None, dropout_p=0.0
        )
        q_18 = k_18 = v_18 = None
        transpose_19 = x_239.transpose(1, 2)
        x_239 = None
        x_240 = transpose_19.reshape(1, 196, 1408)
        transpose_19 = None
        x_241 = torch._C._nn.linear(
            x_240,
            l_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_bias_,
        )
        x_240 = l_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_242 = torch.nn.functional.dropout(x_241, 0.0, False, False)
        x_241 = None
        x_243 = x_237 + x_242
        x_237 = x_242 = None
        x_244 = torch.nn.functional.layer_norm(
            x_243,
            (1408,),
            l_self_modules_blocks_modules_18_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_18_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_18_modules_norm2_parameters_bias_
        ) = None
        x_245 = torch._C._nn.linear(
            x_244,
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_244 = (
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_246 = torch._C._nn.gelu(x_245, approximate="none")
        x_245 = None
        x_247 = torch.nn.functional.dropout(x_246, 0.0, False, False)
        x_246 = None
        x_248 = torch._C._nn.linear(
            x_247,
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_247 = (
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_249 = torch.nn.functional.dropout(x_248, 0.0, False, False)
        x_248 = None
        x_250 = x_243 + x_249
        x_243 = x_249 = None
        x_251 = torch.nn.functional.layer_norm(
            x_250,
            (1408,),
            l_self_modules_blocks_modules_19_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_19_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_19_modules_norm1_parameters_bias_
        ) = None
        linear_76 = torch._C._nn.linear(
            x_251,
            l_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_bias_,
        )
        x_251 = (
            l_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_38 = linear_76.reshape(1, 196, 3, 16, 88)
        linear_76 = None
        qkv_19 = reshape_38.permute(2, 0, 3, 1, 4)
        reshape_38 = None
        unbind_19 = qkv_19.unbind(0)
        qkv_19 = None
        q_19 = unbind_19[0]
        k_19 = unbind_19[1]
        v_19 = unbind_19[2]
        unbind_19 = None
        x_252 = torch._C._nn.scaled_dot_product_attention(
            q_19, k_19, v_19, attn_mask=None, dropout_p=0.0
        )
        q_19 = k_19 = v_19 = None
        transpose_20 = x_252.transpose(1, 2)
        x_252 = None
        x_253 = transpose_20.reshape(1, 196, 1408)
        transpose_20 = None
        x_254 = torch._C._nn.linear(
            x_253,
            l_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_bias_,
        )
        x_253 = l_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_255 = torch.nn.functional.dropout(x_254, 0.0, False, False)
        x_254 = None
        x_256 = x_250 + x_255
        x_250 = x_255 = None
        x_257 = torch.nn.functional.layer_norm(
            x_256,
            (1408,),
            l_self_modules_blocks_modules_19_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_19_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_19_modules_norm2_parameters_bias_
        ) = None
        x_258 = torch._C._nn.linear(
            x_257,
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_257 = (
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_259 = torch._C._nn.gelu(x_258, approximate="none")
        x_258 = None
        x_260 = torch.nn.functional.dropout(x_259, 0.0, False, False)
        x_259 = None
        x_261 = torch._C._nn.linear(
            x_260,
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_260 = (
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_262 = torch.nn.functional.dropout(x_261, 0.0, False, False)
        x_261 = None
        x_263 = x_256 + x_262
        x_256 = x_262 = None
        x_264 = torch.nn.functional.layer_norm(
            x_263,
            (1408,),
            l_self_modules_blocks_modules_20_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_20_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_20_modules_norm1_parameters_bias_
        ) = None
        linear_80 = torch._C._nn.linear(
            x_264,
            l_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_bias_,
        )
        x_264 = (
            l_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_40 = linear_80.reshape(1, 196, 3, 16, 88)
        linear_80 = None
        qkv_20 = reshape_40.permute(2, 0, 3, 1, 4)
        reshape_40 = None
        unbind_20 = qkv_20.unbind(0)
        qkv_20 = None
        q_20 = unbind_20[0]
        k_20 = unbind_20[1]
        v_20 = unbind_20[2]
        unbind_20 = None
        x_265 = torch._C._nn.scaled_dot_product_attention(
            q_20, k_20, v_20, attn_mask=None, dropout_p=0.0
        )
        q_20 = k_20 = v_20 = None
        transpose_21 = x_265.transpose(1, 2)
        x_265 = None
        x_266 = transpose_21.reshape(1, 196, 1408)
        transpose_21 = None
        x_267 = torch._C._nn.linear(
            x_266,
            l_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_bias_,
        )
        x_266 = l_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_268 = torch.nn.functional.dropout(x_267, 0.0, False, False)
        x_267 = None
        x_269 = x_263 + x_268
        x_263 = x_268 = None
        x_270 = torch.nn.functional.layer_norm(
            x_269,
            (1408,),
            l_self_modules_blocks_modules_20_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_20_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_20_modules_norm2_parameters_bias_
        ) = None
        x_271 = torch._C._nn.linear(
            x_270,
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_270 = (
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_272 = torch._C._nn.gelu(x_271, approximate="none")
        x_271 = None
        x_273 = torch.nn.functional.dropout(x_272, 0.0, False, False)
        x_272 = None
        x_274 = torch._C._nn.linear(
            x_273,
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_273 = (
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_275 = torch.nn.functional.dropout(x_274, 0.0, False, False)
        x_274 = None
        x_276 = x_269 + x_275
        x_269 = x_275 = None
        x_277 = torch.nn.functional.layer_norm(
            x_276,
            (1408,),
            l_self_modules_blocks_modules_21_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_21_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_21_modules_norm1_parameters_bias_
        ) = None
        linear_84 = torch._C._nn.linear(
            x_277,
            l_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_bias_,
        )
        x_277 = (
            l_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_42 = linear_84.reshape(1, 196, 3, 16, 88)
        linear_84 = None
        qkv_21 = reshape_42.permute(2, 0, 3, 1, 4)
        reshape_42 = None
        unbind_21 = qkv_21.unbind(0)
        qkv_21 = None
        q_21 = unbind_21[0]
        k_21 = unbind_21[1]
        v_21 = unbind_21[2]
        unbind_21 = None
        x_278 = torch._C._nn.scaled_dot_product_attention(
            q_21, k_21, v_21, attn_mask=None, dropout_p=0.0
        )
        q_21 = k_21 = v_21 = None
        transpose_22 = x_278.transpose(1, 2)
        x_278 = None
        x_279 = transpose_22.reshape(1, 196, 1408)
        transpose_22 = None
        x_280 = torch._C._nn.linear(
            x_279,
            l_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_bias_,
        )
        x_279 = l_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_281 = torch.nn.functional.dropout(x_280, 0.0, False, False)
        x_280 = None
        x_282 = x_276 + x_281
        x_276 = x_281 = None
        x_283 = torch.nn.functional.layer_norm(
            x_282,
            (1408,),
            l_self_modules_blocks_modules_21_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_21_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_21_modules_norm2_parameters_bias_
        ) = None
        x_284 = torch._C._nn.linear(
            x_283,
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_283 = (
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_285 = torch._C._nn.gelu(x_284, approximate="none")
        x_284 = None
        x_286 = torch.nn.functional.dropout(x_285, 0.0, False, False)
        x_285 = None
        x_287 = torch._C._nn.linear(
            x_286,
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_286 = (
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_288 = torch.nn.functional.dropout(x_287, 0.0, False, False)
        x_287 = None
        x_289 = x_282 + x_288
        x_282 = x_288 = None
        x_290 = torch.nn.functional.layer_norm(
            x_289,
            (1408,),
            l_self_modules_blocks_modules_22_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_22_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_22_modules_norm1_parameters_bias_
        ) = None
        linear_88 = torch._C._nn.linear(
            x_290,
            l_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_bias_,
        )
        x_290 = (
            l_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_44 = linear_88.reshape(1, 196, 3, 16, 88)
        linear_88 = None
        qkv_22 = reshape_44.permute(2, 0, 3, 1, 4)
        reshape_44 = None
        unbind_22 = qkv_22.unbind(0)
        qkv_22 = None
        q_22 = unbind_22[0]
        k_22 = unbind_22[1]
        v_22 = unbind_22[2]
        unbind_22 = None
        x_291 = torch._C._nn.scaled_dot_product_attention(
            q_22, k_22, v_22, attn_mask=None, dropout_p=0.0
        )
        q_22 = k_22 = v_22 = None
        transpose_23 = x_291.transpose(1, 2)
        x_291 = None
        x_292 = transpose_23.reshape(1, 196, 1408)
        transpose_23 = None
        x_293 = torch._C._nn.linear(
            x_292,
            l_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_bias_,
        )
        x_292 = l_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_294 = torch.nn.functional.dropout(x_293, 0.0, False, False)
        x_293 = None
        x_295 = x_289 + x_294
        x_289 = x_294 = None
        x_296 = torch.nn.functional.layer_norm(
            x_295,
            (1408,),
            l_self_modules_blocks_modules_22_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_22_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_22_modules_norm2_parameters_bias_
        ) = None
        x_297 = torch._C._nn.linear(
            x_296,
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_296 = (
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_298 = torch._C._nn.gelu(x_297, approximate="none")
        x_297 = None
        x_299 = torch.nn.functional.dropout(x_298, 0.0, False, False)
        x_298 = None
        x_300 = torch._C._nn.linear(
            x_299,
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_299 = (
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_301 = torch.nn.functional.dropout(x_300, 0.0, False, False)
        x_300 = None
        x_302 = x_295 + x_301
        x_295 = x_301 = None
        x_303 = torch.nn.functional.layer_norm(
            x_302,
            (1408,),
            l_self_modules_blocks_modules_23_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_23_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_23_modules_norm1_parameters_bias_
        ) = None
        linear_92 = torch._C._nn.linear(
            x_303,
            l_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_bias_,
        )
        x_303 = (
            l_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_46 = linear_92.reshape(1, 196, 3, 16, 88)
        linear_92 = None
        qkv_23 = reshape_46.permute(2, 0, 3, 1, 4)
        reshape_46 = None
        unbind_23 = qkv_23.unbind(0)
        qkv_23 = None
        q_23 = unbind_23[0]
        k_23 = unbind_23[1]
        v_23 = unbind_23[2]
        unbind_23 = None
        x_304 = torch._C._nn.scaled_dot_product_attention(
            q_23, k_23, v_23, attn_mask=None, dropout_p=0.0
        )
        q_23 = k_23 = v_23 = None
        transpose_24 = x_304.transpose(1, 2)
        x_304 = None
        x_305 = transpose_24.reshape(1, 196, 1408)
        transpose_24 = None
        x_306 = torch._C._nn.linear(
            x_305,
            l_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_bias_,
        )
        x_305 = l_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_307 = torch.nn.functional.dropout(x_306, 0.0, False, False)
        x_306 = None
        x_308 = x_302 + x_307
        x_302 = x_307 = None
        x_309 = torch.nn.functional.layer_norm(
            x_308,
            (1408,),
            l_self_modules_blocks_modules_23_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_23_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_23_modules_norm2_parameters_bias_
        ) = None
        x_310 = torch._C._nn.linear(
            x_309,
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_309 = (
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_311 = torch._C._nn.gelu(x_310, approximate="none")
        x_310 = None
        x_312 = torch.nn.functional.dropout(x_311, 0.0, False, False)
        x_311 = None
        x_313 = torch._C._nn.linear(
            x_312,
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_312 = (
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_314 = torch.nn.functional.dropout(x_313, 0.0, False, False)
        x_313 = None
        x_315 = x_308 + x_314
        x_308 = x_314 = None
        x_316 = torch.nn.functional.layer_norm(
            x_315,
            (1408,),
            l_self_modules_blocks_modules_24_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_24_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_24_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_24_modules_norm1_parameters_bias_
        ) = None
        linear_96 = torch._C._nn.linear(
            x_316,
            l_self_modules_blocks_modules_24_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_24_modules_attn_modules_qkv_parameters_bias_,
        )
        x_316 = (
            l_self_modules_blocks_modules_24_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_24_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_48 = linear_96.reshape(1, 196, 3, 16, 88)
        linear_96 = None
        qkv_24 = reshape_48.permute(2, 0, 3, 1, 4)
        reshape_48 = None
        unbind_24 = qkv_24.unbind(0)
        qkv_24 = None
        q_24 = unbind_24[0]
        k_24 = unbind_24[1]
        v_24 = unbind_24[2]
        unbind_24 = None
        x_317 = torch._C._nn.scaled_dot_product_attention(
            q_24, k_24, v_24, attn_mask=None, dropout_p=0.0
        )
        q_24 = k_24 = v_24 = None
        transpose_25 = x_317.transpose(1, 2)
        x_317 = None
        x_318 = transpose_25.reshape(1, 196, 1408)
        transpose_25 = None
        x_319 = torch._C._nn.linear(
            x_318,
            l_self_modules_blocks_modules_24_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_24_modules_attn_modules_proj_parameters_bias_,
        )
        x_318 = l_self_modules_blocks_modules_24_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_24_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_320 = torch.nn.functional.dropout(x_319, 0.0, False, False)
        x_319 = None
        x_321 = x_315 + x_320
        x_315 = x_320 = None
        x_322 = torch.nn.functional.layer_norm(
            x_321,
            (1408,),
            l_self_modules_blocks_modules_24_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_24_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_24_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_24_modules_norm2_parameters_bias_
        ) = None
        x_323 = torch._C._nn.linear(
            x_322,
            l_self_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_322 = (
            l_self_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_324 = torch._C._nn.gelu(x_323, approximate="none")
        x_323 = None
        x_325 = torch.nn.functional.dropout(x_324, 0.0, False, False)
        x_324 = None
        x_326 = torch._C._nn.linear(
            x_325,
            l_self_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_325 = (
            l_self_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_327 = torch.nn.functional.dropout(x_326, 0.0, False, False)
        x_326 = None
        x_328 = x_321 + x_327
        x_321 = x_327 = None
        x_329 = torch.nn.functional.layer_norm(
            x_328,
            (1408,),
            l_self_modules_blocks_modules_25_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_25_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_25_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_25_modules_norm1_parameters_bias_
        ) = None
        linear_100 = torch._C._nn.linear(
            x_329,
            l_self_modules_blocks_modules_25_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_25_modules_attn_modules_qkv_parameters_bias_,
        )
        x_329 = (
            l_self_modules_blocks_modules_25_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_25_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_50 = linear_100.reshape(1, 196, 3, 16, 88)
        linear_100 = None
        qkv_25 = reshape_50.permute(2, 0, 3, 1, 4)
        reshape_50 = None
        unbind_25 = qkv_25.unbind(0)
        qkv_25 = None
        q_25 = unbind_25[0]
        k_25 = unbind_25[1]
        v_25 = unbind_25[2]
        unbind_25 = None
        x_330 = torch._C._nn.scaled_dot_product_attention(
            q_25, k_25, v_25, attn_mask=None, dropout_p=0.0
        )
        q_25 = k_25 = v_25 = None
        transpose_26 = x_330.transpose(1, 2)
        x_330 = None
        x_331 = transpose_26.reshape(1, 196, 1408)
        transpose_26 = None
        x_332 = torch._C._nn.linear(
            x_331,
            l_self_modules_blocks_modules_25_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_25_modules_attn_modules_proj_parameters_bias_,
        )
        x_331 = l_self_modules_blocks_modules_25_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_25_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_333 = torch.nn.functional.dropout(x_332, 0.0, False, False)
        x_332 = None
        x_334 = x_328 + x_333
        x_328 = x_333 = None
        x_335 = torch.nn.functional.layer_norm(
            x_334,
            (1408,),
            l_self_modules_blocks_modules_25_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_25_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_25_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_25_modules_norm2_parameters_bias_
        ) = None
        x_336 = torch._C._nn.linear(
            x_335,
            l_self_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_335 = (
            l_self_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_337 = torch._C._nn.gelu(x_336, approximate="none")
        x_336 = None
        x_338 = torch.nn.functional.dropout(x_337, 0.0, False, False)
        x_337 = None
        x_339 = torch._C._nn.linear(
            x_338,
            l_self_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_338 = (
            l_self_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_340 = torch.nn.functional.dropout(x_339, 0.0, False, False)
        x_339 = None
        x_341 = x_334 + x_340
        x_334 = x_340 = None
        x_342 = torch.nn.functional.layer_norm(
            x_341,
            (1408,),
            l_self_modules_blocks_modules_26_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_26_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_26_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_26_modules_norm1_parameters_bias_
        ) = None
        linear_104 = torch._C._nn.linear(
            x_342,
            l_self_modules_blocks_modules_26_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_26_modules_attn_modules_qkv_parameters_bias_,
        )
        x_342 = (
            l_self_modules_blocks_modules_26_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_26_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_52 = linear_104.reshape(1, 196, 3, 16, 88)
        linear_104 = None
        qkv_26 = reshape_52.permute(2, 0, 3, 1, 4)
        reshape_52 = None
        unbind_26 = qkv_26.unbind(0)
        qkv_26 = None
        q_26 = unbind_26[0]
        k_26 = unbind_26[1]
        v_26 = unbind_26[2]
        unbind_26 = None
        x_343 = torch._C._nn.scaled_dot_product_attention(
            q_26, k_26, v_26, attn_mask=None, dropout_p=0.0
        )
        q_26 = k_26 = v_26 = None
        transpose_27 = x_343.transpose(1, 2)
        x_343 = None
        x_344 = transpose_27.reshape(1, 196, 1408)
        transpose_27 = None
        x_345 = torch._C._nn.linear(
            x_344,
            l_self_modules_blocks_modules_26_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_26_modules_attn_modules_proj_parameters_bias_,
        )
        x_344 = l_self_modules_blocks_modules_26_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_26_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_346 = torch.nn.functional.dropout(x_345, 0.0, False, False)
        x_345 = None
        x_347 = x_341 + x_346
        x_341 = x_346 = None
        x_348 = torch.nn.functional.layer_norm(
            x_347,
            (1408,),
            l_self_modules_blocks_modules_26_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_26_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_26_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_26_modules_norm2_parameters_bias_
        ) = None
        x_349 = torch._C._nn.linear(
            x_348,
            l_self_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_348 = (
            l_self_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_350 = torch._C._nn.gelu(x_349, approximate="none")
        x_349 = None
        x_351 = torch.nn.functional.dropout(x_350, 0.0, False, False)
        x_350 = None
        x_352 = torch._C._nn.linear(
            x_351,
            l_self_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_351 = (
            l_self_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_353 = torch.nn.functional.dropout(x_352, 0.0, False, False)
        x_352 = None
        x_354 = x_347 + x_353
        x_347 = x_353 = None
        x_355 = torch.nn.functional.layer_norm(
            x_354,
            (1408,),
            l_self_modules_blocks_modules_27_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_27_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_27_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_27_modules_norm1_parameters_bias_
        ) = None
        linear_108 = torch._C._nn.linear(
            x_355,
            l_self_modules_blocks_modules_27_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_27_modules_attn_modules_qkv_parameters_bias_,
        )
        x_355 = (
            l_self_modules_blocks_modules_27_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_27_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_54 = linear_108.reshape(1, 196, 3, 16, 88)
        linear_108 = None
        qkv_27 = reshape_54.permute(2, 0, 3, 1, 4)
        reshape_54 = None
        unbind_27 = qkv_27.unbind(0)
        qkv_27 = None
        q_27 = unbind_27[0]
        k_27 = unbind_27[1]
        v_27 = unbind_27[2]
        unbind_27 = None
        x_356 = torch._C._nn.scaled_dot_product_attention(
            q_27, k_27, v_27, attn_mask=None, dropout_p=0.0
        )
        q_27 = k_27 = v_27 = None
        transpose_28 = x_356.transpose(1, 2)
        x_356 = None
        x_357 = transpose_28.reshape(1, 196, 1408)
        transpose_28 = None
        x_358 = torch._C._nn.linear(
            x_357,
            l_self_modules_blocks_modules_27_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_27_modules_attn_modules_proj_parameters_bias_,
        )
        x_357 = l_self_modules_blocks_modules_27_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_27_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_359 = torch.nn.functional.dropout(x_358, 0.0, False, False)
        x_358 = None
        x_360 = x_354 + x_359
        x_354 = x_359 = None
        x_361 = torch.nn.functional.layer_norm(
            x_360,
            (1408,),
            l_self_modules_blocks_modules_27_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_27_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_27_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_27_modules_norm2_parameters_bias_
        ) = None
        x_362 = torch._C._nn.linear(
            x_361,
            l_self_modules_blocks_modules_27_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_27_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_361 = (
            l_self_modules_blocks_modules_27_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_27_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_363 = torch._C._nn.gelu(x_362, approximate="none")
        x_362 = None
        x_364 = torch.nn.functional.dropout(x_363, 0.0, False, False)
        x_363 = None
        x_365 = torch._C._nn.linear(
            x_364,
            l_self_modules_blocks_modules_27_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_27_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_364 = (
            l_self_modules_blocks_modules_27_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_27_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_366 = torch.nn.functional.dropout(x_365, 0.0, False, False)
        x_365 = None
        x_367 = x_360 + x_366
        x_360 = x_366 = None
        x_368 = torch.nn.functional.layer_norm(
            x_367,
            (1408,),
            l_self_modules_blocks_modules_28_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_28_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_28_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_28_modules_norm1_parameters_bias_
        ) = None
        linear_112 = torch._C._nn.linear(
            x_368,
            l_self_modules_blocks_modules_28_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_28_modules_attn_modules_qkv_parameters_bias_,
        )
        x_368 = (
            l_self_modules_blocks_modules_28_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_28_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_56 = linear_112.reshape(1, 196, 3, 16, 88)
        linear_112 = None
        qkv_28 = reshape_56.permute(2, 0, 3, 1, 4)
        reshape_56 = None
        unbind_28 = qkv_28.unbind(0)
        qkv_28 = None
        q_28 = unbind_28[0]
        k_28 = unbind_28[1]
        v_28 = unbind_28[2]
        unbind_28 = None
        x_369 = torch._C._nn.scaled_dot_product_attention(
            q_28, k_28, v_28, attn_mask=None, dropout_p=0.0
        )
        q_28 = k_28 = v_28 = None
        transpose_29 = x_369.transpose(1, 2)
        x_369 = None
        x_370 = transpose_29.reshape(1, 196, 1408)
        transpose_29 = None
        x_371 = torch._C._nn.linear(
            x_370,
            l_self_modules_blocks_modules_28_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_28_modules_attn_modules_proj_parameters_bias_,
        )
        x_370 = l_self_modules_blocks_modules_28_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_28_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_372 = torch.nn.functional.dropout(x_371, 0.0, False, False)
        x_371 = None
        x_373 = x_367 + x_372
        x_367 = x_372 = None
        x_374 = torch.nn.functional.layer_norm(
            x_373,
            (1408,),
            l_self_modules_blocks_modules_28_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_28_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_28_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_28_modules_norm2_parameters_bias_
        ) = None
        x_375 = torch._C._nn.linear(
            x_374,
            l_self_modules_blocks_modules_28_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_28_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_374 = (
            l_self_modules_blocks_modules_28_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_28_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_376 = torch._C._nn.gelu(x_375, approximate="none")
        x_375 = None
        x_377 = torch.nn.functional.dropout(x_376, 0.0, False, False)
        x_376 = None
        x_378 = torch._C._nn.linear(
            x_377,
            l_self_modules_blocks_modules_28_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_28_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_377 = (
            l_self_modules_blocks_modules_28_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_28_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_379 = torch.nn.functional.dropout(x_378, 0.0, False, False)
        x_378 = None
        x_380 = x_373 + x_379
        x_373 = x_379 = None
        x_381 = torch.nn.functional.layer_norm(
            x_380,
            (1408,),
            l_self_modules_blocks_modules_29_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_29_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_29_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_29_modules_norm1_parameters_bias_
        ) = None
        linear_116 = torch._C._nn.linear(
            x_381,
            l_self_modules_blocks_modules_29_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_29_modules_attn_modules_qkv_parameters_bias_,
        )
        x_381 = (
            l_self_modules_blocks_modules_29_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_29_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_58 = linear_116.reshape(1, 196, 3, 16, 88)
        linear_116 = None
        qkv_29 = reshape_58.permute(2, 0, 3, 1, 4)
        reshape_58 = None
        unbind_29 = qkv_29.unbind(0)
        qkv_29 = None
        q_29 = unbind_29[0]
        k_29 = unbind_29[1]
        v_29 = unbind_29[2]
        unbind_29 = None
        x_382 = torch._C._nn.scaled_dot_product_attention(
            q_29, k_29, v_29, attn_mask=None, dropout_p=0.0
        )
        q_29 = k_29 = v_29 = None
        transpose_30 = x_382.transpose(1, 2)
        x_382 = None
        x_383 = transpose_30.reshape(1, 196, 1408)
        transpose_30 = None
        x_384 = torch._C._nn.linear(
            x_383,
            l_self_modules_blocks_modules_29_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_29_modules_attn_modules_proj_parameters_bias_,
        )
        x_383 = l_self_modules_blocks_modules_29_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_29_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_385 = torch.nn.functional.dropout(x_384, 0.0, False, False)
        x_384 = None
        x_386 = x_380 + x_385
        x_380 = x_385 = None
        x_387 = torch.nn.functional.layer_norm(
            x_386,
            (1408,),
            l_self_modules_blocks_modules_29_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_29_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_29_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_29_modules_norm2_parameters_bias_
        ) = None
        x_388 = torch._C._nn.linear(
            x_387,
            l_self_modules_blocks_modules_29_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_29_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_387 = (
            l_self_modules_blocks_modules_29_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_29_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_389 = torch._C._nn.gelu(x_388, approximate="none")
        x_388 = None
        x_390 = torch.nn.functional.dropout(x_389, 0.0, False, False)
        x_389 = None
        x_391 = torch._C._nn.linear(
            x_390,
            l_self_modules_blocks_modules_29_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_29_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_390 = (
            l_self_modules_blocks_modules_29_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_29_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_392 = torch.nn.functional.dropout(x_391, 0.0, False, False)
        x_391 = None
        x_393 = x_386 + x_392
        x_386 = x_392 = None
        x_394 = torch.nn.functional.layer_norm(
            x_393,
            (1408,),
            l_self_modules_blocks_modules_30_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_30_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_30_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_30_modules_norm1_parameters_bias_
        ) = None
        linear_120 = torch._C._nn.linear(
            x_394,
            l_self_modules_blocks_modules_30_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_30_modules_attn_modules_qkv_parameters_bias_,
        )
        x_394 = (
            l_self_modules_blocks_modules_30_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_30_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_60 = linear_120.reshape(1, 196, 3, 16, 88)
        linear_120 = None
        qkv_30 = reshape_60.permute(2, 0, 3, 1, 4)
        reshape_60 = None
        unbind_30 = qkv_30.unbind(0)
        qkv_30 = None
        q_30 = unbind_30[0]
        k_30 = unbind_30[1]
        v_30 = unbind_30[2]
        unbind_30 = None
        x_395 = torch._C._nn.scaled_dot_product_attention(
            q_30, k_30, v_30, attn_mask=None, dropout_p=0.0
        )
        q_30 = k_30 = v_30 = None
        transpose_31 = x_395.transpose(1, 2)
        x_395 = None
        x_396 = transpose_31.reshape(1, 196, 1408)
        transpose_31 = None
        x_397 = torch._C._nn.linear(
            x_396,
            l_self_modules_blocks_modules_30_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_30_modules_attn_modules_proj_parameters_bias_,
        )
        x_396 = l_self_modules_blocks_modules_30_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_30_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_398 = torch.nn.functional.dropout(x_397, 0.0, False, False)
        x_397 = None
        x_399 = x_393 + x_398
        x_393 = x_398 = None
        x_400 = torch.nn.functional.layer_norm(
            x_399,
            (1408,),
            l_self_modules_blocks_modules_30_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_30_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_30_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_30_modules_norm2_parameters_bias_
        ) = None
        x_401 = torch._C._nn.linear(
            x_400,
            l_self_modules_blocks_modules_30_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_30_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_400 = (
            l_self_modules_blocks_modules_30_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_30_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_402 = torch._C._nn.gelu(x_401, approximate="none")
        x_401 = None
        x_403 = torch.nn.functional.dropout(x_402, 0.0, False, False)
        x_402 = None
        x_404 = torch._C._nn.linear(
            x_403,
            l_self_modules_blocks_modules_30_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_30_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_403 = (
            l_self_modules_blocks_modules_30_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_30_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_405 = torch.nn.functional.dropout(x_404, 0.0, False, False)
        x_404 = None
        x_406 = x_399 + x_405
        x_399 = x_405 = None
        x_407 = torch.nn.functional.layer_norm(
            x_406,
            (1408,),
            l_self_modules_blocks_modules_31_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_31_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_31_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_31_modules_norm1_parameters_bias_
        ) = None
        linear_124 = torch._C._nn.linear(
            x_407,
            l_self_modules_blocks_modules_31_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_31_modules_attn_modules_qkv_parameters_bias_,
        )
        x_407 = (
            l_self_modules_blocks_modules_31_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_31_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_62 = linear_124.reshape(1, 196, 3, 16, 88)
        linear_124 = None
        qkv_31 = reshape_62.permute(2, 0, 3, 1, 4)
        reshape_62 = None
        unbind_31 = qkv_31.unbind(0)
        qkv_31 = None
        q_31 = unbind_31[0]
        k_31 = unbind_31[1]
        v_31 = unbind_31[2]
        unbind_31 = None
        x_408 = torch._C._nn.scaled_dot_product_attention(
            q_31, k_31, v_31, attn_mask=None, dropout_p=0.0
        )
        q_31 = k_31 = v_31 = None
        transpose_32 = x_408.transpose(1, 2)
        x_408 = None
        x_409 = transpose_32.reshape(1, 196, 1408)
        transpose_32 = None
        x_410 = torch._C._nn.linear(
            x_409,
            l_self_modules_blocks_modules_31_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_31_modules_attn_modules_proj_parameters_bias_,
        )
        x_409 = l_self_modules_blocks_modules_31_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_31_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_411 = torch.nn.functional.dropout(x_410, 0.0, False, False)
        x_410 = None
        x_412 = x_406 + x_411
        x_406 = x_411 = None
        x_413 = torch.nn.functional.layer_norm(
            x_412,
            (1408,),
            l_self_modules_blocks_modules_31_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_31_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_31_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_31_modules_norm2_parameters_bias_
        ) = None
        x_414 = torch._C._nn.linear(
            x_413,
            l_self_modules_blocks_modules_31_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_31_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_413 = (
            l_self_modules_blocks_modules_31_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_31_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_415 = torch._C._nn.gelu(x_414, approximate="none")
        x_414 = None
        x_416 = torch.nn.functional.dropout(x_415, 0.0, False, False)
        x_415 = None
        x_417 = torch._C._nn.linear(
            x_416,
            l_self_modules_blocks_modules_31_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_31_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_416 = (
            l_self_modules_blocks_modules_31_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_31_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_418 = torch.nn.functional.dropout(x_417, 0.0, False, False)
        x_417 = None
        x_419 = x_412 + x_418
        x_412 = x_418 = None
        x_420 = torch.nn.functional.layer_norm(
            x_419,
            (1408,),
            l_self_modules_blocks_modules_32_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_32_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_32_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_32_modules_norm1_parameters_bias_
        ) = None
        linear_128 = torch._C._nn.linear(
            x_420,
            l_self_modules_blocks_modules_32_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_32_modules_attn_modules_qkv_parameters_bias_,
        )
        x_420 = (
            l_self_modules_blocks_modules_32_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_32_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_64 = linear_128.reshape(1, 196, 3, 16, 88)
        linear_128 = None
        qkv_32 = reshape_64.permute(2, 0, 3, 1, 4)
        reshape_64 = None
        unbind_32 = qkv_32.unbind(0)
        qkv_32 = None
        q_32 = unbind_32[0]
        k_32 = unbind_32[1]
        v_32 = unbind_32[2]
        unbind_32 = None
        x_421 = torch._C._nn.scaled_dot_product_attention(
            q_32, k_32, v_32, attn_mask=None, dropout_p=0.0
        )
        q_32 = k_32 = v_32 = None
        transpose_33 = x_421.transpose(1, 2)
        x_421 = None
        x_422 = transpose_33.reshape(1, 196, 1408)
        transpose_33 = None
        x_423 = torch._C._nn.linear(
            x_422,
            l_self_modules_blocks_modules_32_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_32_modules_attn_modules_proj_parameters_bias_,
        )
        x_422 = l_self_modules_blocks_modules_32_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_32_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_424 = torch.nn.functional.dropout(x_423, 0.0, False, False)
        x_423 = None
        x_425 = x_419 + x_424
        x_419 = x_424 = None
        x_426 = torch.nn.functional.layer_norm(
            x_425,
            (1408,),
            l_self_modules_blocks_modules_32_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_32_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_32_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_32_modules_norm2_parameters_bias_
        ) = None
        x_427 = torch._C._nn.linear(
            x_426,
            l_self_modules_blocks_modules_32_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_32_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_426 = (
            l_self_modules_blocks_modules_32_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_32_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_428 = torch._C._nn.gelu(x_427, approximate="none")
        x_427 = None
        x_429 = torch.nn.functional.dropout(x_428, 0.0, False, False)
        x_428 = None
        x_430 = torch._C._nn.linear(
            x_429,
            l_self_modules_blocks_modules_32_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_32_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_429 = (
            l_self_modules_blocks_modules_32_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_32_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_431 = torch.nn.functional.dropout(x_430, 0.0, False, False)
        x_430 = None
        x_432 = x_425 + x_431
        x_425 = x_431 = None
        x_433 = torch.nn.functional.layer_norm(
            x_432,
            (1408,),
            l_self_modules_blocks_modules_33_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_33_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_33_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_33_modules_norm1_parameters_bias_
        ) = None
        linear_132 = torch._C._nn.linear(
            x_433,
            l_self_modules_blocks_modules_33_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_33_modules_attn_modules_qkv_parameters_bias_,
        )
        x_433 = (
            l_self_modules_blocks_modules_33_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_33_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_66 = linear_132.reshape(1, 196, 3, 16, 88)
        linear_132 = None
        qkv_33 = reshape_66.permute(2, 0, 3, 1, 4)
        reshape_66 = None
        unbind_33 = qkv_33.unbind(0)
        qkv_33 = None
        q_33 = unbind_33[0]
        k_33 = unbind_33[1]
        v_33 = unbind_33[2]
        unbind_33 = None
        x_434 = torch._C._nn.scaled_dot_product_attention(
            q_33, k_33, v_33, attn_mask=None, dropout_p=0.0
        )
        q_33 = k_33 = v_33 = None
        transpose_34 = x_434.transpose(1, 2)
        x_434 = None
        x_435 = transpose_34.reshape(1, 196, 1408)
        transpose_34 = None
        x_436 = torch._C._nn.linear(
            x_435,
            l_self_modules_blocks_modules_33_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_33_modules_attn_modules_proj_parameters_bias_,
        )
        x_435 = l_self_modules_blocks_modules_33_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_33_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_437 = torch.nn.functional.dropout(x_436, 0.0, False, False)
        x_436 = None
        x_438 = x_432 + x_437
        x_432 = x_437 = None
        x_439 = torch.nn.functional.layer_norm(
            x_438,
            (1408,),
            l_self_modules_blocks_modules_33_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_33_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_33_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_33_modules_norm2_parameters_bias_
        ) = None
        x_440 = torch._C._nn.linear(
            x_439,
            l_self_modules_blocks_modules_33_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_33_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_439 = (
            l_self_modules_blocks_modules_33_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_33_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_441 = torch._C._nn.gelu(x_440, approximate="none")
        x_440 = None
        x_442 = torch.nn.functional.dropout(x_441, 0.0, False, False)
        x_441 = None
        x_443 = torch._C._nn.linear(
            x_442,
            l_self_modules_blocks_modules_33_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_33_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_442 = (
            l_self_modules_blocks_modules_33_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_33_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_444 = torch.nn.functional.dropout(x_443, 0.0, False, False)
        x_443 = None
        x_445 = x_438 + x_444
        x_438 = x_444 = None
        x_446 = torch.nn.functional.layer_norm(
            x_445,
            (1408,),
            l_self_modules_blocks_modules_34_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_34_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_34_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_34_modules_norm1_parameters_bias_
        ) = None
        linear_136 = torch._C._nn.linear(
            x_446,
            l_self_modules_blocks_modules_34_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_34_modules_attn_modules_qkv_parameters_bias_,
        )
        x_446 = (
            l_self_modules_blocks_modules_34_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_34_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_68 = linear_136.reshape(1, 196, 3, 16, 88)
        linear_136 = None
        qkv_34 = reshape_68.permute(2, 0, 3, 1, 4)
        reshape_68 = None
        unbind_34 = qkv_34.unbind(0)
        qkv_34 = None
        q_34 = unbind_34[0]
        k_34 = unbind_34[1]
        v_34 = unbind_34[2]
        unbind_34 = None
        x_447 = torch._C._nn.scaled_dot_product_attention(
            q_34, k_34, v_34, attn_mask=None, dropout_p=0.0
        )
        q_34 = k_34 = v_34 = None
        transpose_35 = x_447.transpose(1, 2)
        x_447 = None
        x_448 = transpose_35.reshape(1, 196, 1408)
        transpose_35 = None
        x_449 = torch._C._nn.linear(
            x_448,
            l_self_modules_blocks_modules_34_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_34_modules_attn_modules_proj_parameters_bias_,
        )
        x_448 = l_self_modules_blocks_modules_34_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_34_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_450 = torch.nn.functional.dropout(x_449, 0.0, False, False)
        x_449 = None
        x_451 = x_445 + x_450
        x_445 = x_450 = None
        x_452 = torch.nn.functional.layer_norm(
            x_451,
            (1408,),
            l_self_modules_blocks_modules_34_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_34_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_34_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_34_modules_norm2_parameters_bias_
        ) = None
        x_453 = torch._C._nn.linear(
            x_452,
            l_self_modules_blocks_modules_34_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_34_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_452 = (
            l_self_modules_blocks_modules_34_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_34_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_454 = torch._C._nn.gelu(x_453, approximate="none")
        x_453 = None
        x_455 = torch.nn.functional.dropout(x_454, 0.0, False, False)
        x_454 = None
        x_456 = torch._C._nn.linear(
            x_455,
            l_self_modules_blocks_modules_34_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_34_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_455 = (
            l_self_modules_blocks_modules_34_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_34_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_457 = torch.nn.functional.dropout(x_456, 0.0, False, False)
        x_456 = None
        x_458 = x_451 + x_457
        x_451 = x_457 = None
        x_459 = torch.nn.functional.layer_norm(
            x_458,
            (1408,),
            l_self_modules_blocks_modules_35_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_35_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_35_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_35_modules_norm1_parameters_bias_
        ) = None
        linear_140 = torch._C._nn.linear(
            x_459,
            l_self_modules_blocks_modules_35_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_35_modules_attn_modules_qkv_parameters_bias_,
        )
        x_459 = (
            l_self_modules_blocks_modules_35_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_35_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_70 = linear_140.reshape(1, 196, 3, 16, 88)
        linear_140 = None
        qkv_35 = reshape_70.permute(2, 0, 3, 1, 4)
        reshape_70 = None
        unbind_35 = qkv_35.unbind(0)
        qkv_35 = None
        q_35 = unbind_35[0]
        k_35 = unbind_35[1]
        v_35 = unbind_35[2]
        unbind_35 = None
        x_460 = torch._C._nn.scaled_dot_product_attention(
            q_35, k_35, v_35, attn_mask=None, dropout_p=0.0
        )
        q_35 = k_35 = v_35 = None
        transpose_36 = x_460.transpose(1, 2)
        x_460 = None
        x_461 = transpose_36.reshape(1, 196, 1408)
        transpose_36 = None
        x_462 = torch._C._nn.linear(
            x_461,
            l_self_modules_blocks_modules_35_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_35_modules_attn_modules_proj_parameters_bias_,
        )
        x_461 = l_self_modules_blocks_modules_35_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_35_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_463 = torch.nn.functional.dropout(x_462, 0.0, False, False)
        x_462 = None
        x_464 = x_458 + x_463
        x_458 = x_463 = None
        x_465 = torch.nn.functional.layer_norm(
            x_464,
            (1408,),
            l_self_modules_blocks_modules_35_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_35_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_35_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_35_modules_norm2_parameters_bias_
        ) = None
        x_466 = torch._C._nn.linear(
            x_465,
            l_self_modules_blocks_modules_35_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_35_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_465 = (
            l_self_modules_blocks_modules_35_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_35_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_467 = torch._C._nn.gelu(x_466, approximate="none")
        x_466 = None
        x_468 = torch.nn.functional.dropout(x_467, 0.0, False, False)
        x_467 = None
        x_469 = torch._C._nn.linear(
            x_468,
            l_self_modules_blocks_modules_35_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_35_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_468 = (
            l_self_modules_blocks_modules_35_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_35_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_470 = torch.nn.functional.dropout(x_469, 0.0, False, False)
        x_469 = None
        x_471 = x_464 + x_470
        x_464 = x_470 = None
        x_472 = torch.nn.functional.layer_norm(
            x_471,
            (1408,),
            l_self_modules_blocks_modules_36_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_36_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_36_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_36_modules_norm1_parameters_bias_
        ) = None
        linear_144 = torch._C._nn.linear(
            x_472,
            l_self_modules_blocks_modules_36_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_36_modules_attn_modules_qkv_parameters_bias_,
        )
        x_472 = (
            l_self_modules_blocks_modules_36_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_36_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_72 = linear_144.reshape(1, 196, 3, 16, 88)
        linear_144 = None
        qkv_36 = reshape_72.permute(2, 0, 3, 1, 4)
        reshape_72 = None
        unbind_36 = qkv_36.unbind(0)
        qkv_36 = None
        q_36 = unbind_36[0]
        k_36 = unbind_36[1]
        v_36 = unbind_36[2]
        unbind_36 = None
        x_473 = torch._C._nn.scaled_dot_product_attention(
            q_36, k_36, v_36, attn_mask=None, dropout_p=0.0
        )
        q_36 = k_36 = v_36 = None
        transpose_37 = x_473.transpose(1, 2)
        x_473 = None
        x_474 = transpose_37.reshape(1, 196, 1408)
        transpose_37 = None
        x_475 = torch._C._nn.linear(
            x_474,
            l_self_modules_blocks_modules_36_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_36_modules_attn_modules_proj_parameters_bias_,
        )
        x_474 = l_self_modules_blocks_modules_36_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_36_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_476 = torch.nn.functional.dropout(x_475, 0.0, False, False)
        x_475 = None
        x_477 = x_471 + x_476
        x_471 = x_476 = None
        x_478 = torch.nn.functional.layer_norm(
            x_477,
            (1408,),
            l_self_modules_blocks_modules_36_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_36_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_36_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_36_modules_norm2_parameters_bias_
        ) = None
        x_479 = torch._C._nn.linear(
            x_478,
            l_self_modules_blocks_modules_36_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_36_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_478 = (
            l_self_modules_blocks_modules_36_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_36_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_480 = torch._C._nn.gelu(x_479, approximate="none")
        x_479 = None
        x_481 = torch.nn.functional.dropout(x_480, 0.0, False, False)
        x_480 = None
        x_482 = torch._C._nn.linear(
            x_481,
            l_self_modules_blocks_modules_36_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_36_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_481 = (
            l_self_modules_blocks_modules_36_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_36_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_483 = torch.nn.functional.dropout(x_482, 0.0, False, False)
        x_482 = None
        x_484 = x_477 + x_483
        x_477 = x_483 = None
        x_485 = torch.nn.functional.layer_norm(
            x_484,
            (1408,),
            l_self_modules_blocks_modules_37_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_37_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_37_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_37_modules_norm1_parameters_bias_
        ) = None
        linear_148 = torch._C._nn.linear(
            x_485,
            l_self_modules_blocks_modules_37_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_37_modules_attn_modules_qkv_parameters_bias_,
        )
        x_485 = (
            l_self_modules_blocks_modules_37_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_37_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_74 = linear_148.reshape(1, 196, 3, 16, 88)
        linear_148 = None
        qkv_37 = reshape_74.permute(2, 0, 3, 1, 4)
        reshape_74 = None
        unbind_37 = qkv_37.unbind(0)
        qkv_37 = None
        q_37 = unbind_37[0]
        k_37 = unbind_37[1]
        v_37 = unbind_37[2]
        unbind_37 = None
        x_486 = torch._C._nn.scaled_dot_product_attention(
            q_37, k_37, v_37, attn_mask=None, dropout_p=0.0
        )
        q_37 = k_37 = v_37 = None
        transpose_38 = x_486.transpose(1, 2)
        x_486 = None
        x_487 = transpose_38.reshape(1, 196, 1408)
        transpose_38 = None
        x_488 = torch._C._nn.linear(
            x_487,
            l_self_modules_blocks_modules_37_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_37_modules_attn_modules_proj_parameters_bias_,
        )
        x_487 = l_self_modules_blocks_modules_37_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_37_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_489 = torch.nn.functional.dropout(x_488, 0.0, False, False)
        x_488 = None
        x_490 = x_484 + x_489
        x_484 = x_489 = None
        x_491 = torch.nn.functional.layer_norm(
            x_490,
            (1408,),
            l_self_modules_blocks_modules_37_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_37_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_37_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_37_modules_norm2_parameters_bias_
        ) = None
        x_492 = torch._C._nn.linear(
            x_491,
            l_self_modules_blocks_modules_37_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_37_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_491 = (
            l_self_modules_blocks_modules_37_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_37_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_493 = torch._C._nn.gelu(x_492, approximate="none")
        x_492 = None
        x_494 = torch.nn.functional.dropout(x_493, 0.0, False, False)
        x_493 = None
        x_495 = torch._C._nn.linear(
            x_494,
            l_self_modules_blocks_modules_37_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_37_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_494 = (
            l_self_modules_blocks_modules_37_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_37_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_496 = torch.nn.functional.dropout(x_495, 0.0, False, False)
        x_495 = None
        x_497 = x_490 + x_496
        x_490 = x_496 = None
        x_498 = torch.nn.functional.layer_norm(
            x_497,
            (1408,),
            l_self_modules_blocks_modules_38_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_38_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_38_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_38_modules_norm1_parameters_bias_
        ) = None
        linear_152 = torch._C._nn.linear(
            x_498,
            l_self_modules_blocks_modules_38_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_38_modules_attn_modules_qkv_parameters_bias_,
        )
        x_498 = (
            l_self_modules_blocks_modules_38_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_38_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_76 = linear_152.reshape(1, 196, 3, 16, 88)
        linear_152 = None
        qkv_38 = reshape_76.permute(2, 0, 3, 1, 4)
        reshape_76 = None
        unbind_38 = qkv_38.unbind(0)
        qkv_38 = None
        q_38 = unbind_38[0]
        k_38 = unbind_38[1]
        v_38 = unbind_38[2]
        unbind_38 = None
        x_499 = torch._C._nn.scaled_dot_product_attention(
            q_38, k_38, v_38, attn_mask=None, dropout_p=0.0
        )
        q_38 = k_38 = v_38 = None
        transpose_39 = x_499.transpose(1, 2)
        x_499 = None
        x_500 = transpose_39.reshape(1, 196, 1408)
        transpose_39 = None
        x_501 = torch._C._nn.linear(
            x_500,
            l_self_modules_blocks_modules_38_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_38_modules_attn_modules_proj_parameters_bias_,
        )
        x_500 = l_self_modules_blocks_modules_38_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_38_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_502 = torch.nn.functional.dropout(x_501, 0.0, False, False)
        x_501 = None
        x_503 = x_497 + x_502
        x_497 = x_502 = None
        x_504 = torch.nn.functional.layer_norm(
            x_503,
            (1408,),
            l_self_modules_blocks_modules_38_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_38_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_38_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_38_modules_norm2_parameters_bias_
        ) = None
        x_505 = torch._C._nn.linear(
            x_504,
            l_self_modules_blocks_modules_38_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_38_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_504 = (
            l_self_modules_blocks_modules_38_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_38_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_506 = torch._C._nn.gelu(x_505, approximate="none")
        x_505 = None
        x_507 = torch.nn.functional.dropout(x_506, 0.0, False, False)
        x_506 = None
        x_508 = torch._C._nn.linear(
            x_507,
            l_self_modules_blocks_modules_38_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_38_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_507 = (
            l_self_modules_blocks_modules_38_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_38_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_509 = torch.nn.functional.dropout(x_508, 0.0, False, False)
        x_508 = None
        x_510 = x_503 + x_509
        x_503 = x_509 = None
        x_511 = torch.nn.functional.layer_norm(
            x_510,
            (1408,),
            l_self_modules_blocks_modules_39_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_39_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_39_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_39_modules_norm1_parameters_bias_
        ) = None
        linear_156 = torch._C._nn.linear(
            x_511,
            l_self_modules_blocks_modules_39_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_39_modules_attn_modules_qkv_parameters_bias_,
        )
        x_511 = (
            l_self_modules_blocks_modules_39_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_39_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_78 = linear_156.reshape(1, 196, 3, 16, 88)
        linear_156 = None
        qkv_39 = reshape_78.permute(2, 0, 3, 1, 4)
        reshape_78 = None
        unbind_39 = qkv_39.unbind(0)
        qkv_39 = None
        q_39 = unbind_39[0]
        k_39 = unbind_39[1]
        v_39 = unbind_39[2]
        unbind_39 = None
        x_512 = torch._C._nn.scaled_dot_product_attention(
            q_39, k_39, v_39, attn_mask=None, dropout_p=0.0
        )
        q_39 = k_39 = v_39 = None
        transpose_40 = x_512.transpose(1, 2)
        x_512 = None
        x_513 = transpose_40.reshape(1, 196, 1408)
        transpose_40 = None
        x_514 = torch._C._nn.linear(
            x_513,
            l_self_modules_blocks_modules_39_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_39_modules_attn_modules_proj_parameters_bias_,
        )
        x_513 = l_self_modules_blocks_modules_39_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_39_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_515 = torch.nn.functional.dropout(x_514, 0.0, False, False)
        x_514 = None
        x_516 = x_510 + x_515
        x_510 = x_515 = None
        x_517 = torch.nn.functional.layer_norm(
            x_516,
            (1408,),
            l_self_modules_blocks_modules_39_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_39_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_39_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_39_modules_norm2_parameters_bias_
        ) = None
        x_518 = torch._C._nn.linear(
            x_517,
            l_self_modules_blocks_modules_39_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_39_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_517 = (
            l_self_modules_blocks_modules_39_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_39_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_519 = torch._C._nn.gelu(x_518, approximate="none")
        x_518 = None
        x_520 = torch.nn.functional.dropout(x_519, 0.0, False, False)
        x_519 = None
        x_521 = torch._C._nn.linear(
            x_520,
            l_self_modules_blocks_modules_39_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_39_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_520 = (
            l_self_modules_blocks_modules_39_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_39_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_522 = torch.nn.functional.dropout(x_521, 0.0, False, False)
        x_521 = None
        x_523 = x_516 + x_522
        x_516 = x_522 = None
        x_524 = torch.nn.functional.layer_norm(
            x_523,
            (1408,),
            l_self_modules_norm_parameters_weight_,
            l_self_modules_norm_parameters_bias_,
            1e-06,
        )
        x_523 = (
            l_self_modules_norm_parameters_weight_
        ) = l_self_modules_norm_parameters_bias_ = None
        x_525 = x_524[
            (slice(None, None, None), slice(l_self_num_prefix_tokens, None, None))
        ]
        x_524 = l_self_num_prefix_tokens = None
        x_526 = x_525.mean(dim=1)
        x_525 = None
        x_527 = torch.nn.functional.dropout(x_526, 0.0, False, False)
        x_526 = None
        return (x_527,)
