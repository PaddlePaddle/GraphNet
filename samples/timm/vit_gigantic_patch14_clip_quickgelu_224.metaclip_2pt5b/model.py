import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_x_: torch.Tensor,
        L_self_modules_patch_embed_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_parameters_pos_embed_: torch.nn.parameter.Parameter,
        L_self_parameters_cls_token_: torch.nn.parameter.Parameter,
        L_self_modules_norm_pre_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_norm_pre_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_40_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_40_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_40_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_40_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_40_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_40_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_40_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_40_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_40_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_40_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_40_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_40_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_41_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_41_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_41_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_41_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_41_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_41_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_41_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_41_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_41_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_41_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_41_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_41_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_42_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_42_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_42_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_42_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_42_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_42_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_42_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_42_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_42_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_42_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_42_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_42_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_43_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_43_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_43_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_43_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_43_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_43_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_43_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_43_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_43_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_43_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_43_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_43_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_44_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_44_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_44_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_44_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_44_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_44_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_44_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_44_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_44_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_44_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_44_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_44_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_45_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_45_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_45_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_45_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_45_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_45_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_45_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_45_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_45_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_45_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_45_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_45_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_46_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_46_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_46_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_46_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_46_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_46_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_46_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_46_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_46_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_46_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_46_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_46_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_47_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_47_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_47_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_47_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_47_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_47_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_47_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_47_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_47_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_47_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_47_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_47_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_x_ = L_x_
        l_self_modules_patch_embed_modules_proj_parameters_weight_ = (
            L_self_modules_patch_embed_modules_proj_parameters_weight_
        )
        l_self_parameters_pos_embed_ = L_self_parameters_pos_embed_
        l_self_parameters_cls_token_ = L_self_parameters_cls_token_
        l_self_modules_norm_pre_parameters_weight_ = (
            L_self_modules_norm_pre_parameters_weight_
        )
        l_self_modules_norm_pre_parameters_bias_ = (
            L_self_modules_norm_pre_parameters_bias_
        )
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
        l_self_modules_blocks_modules_40_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_40_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_40_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_40_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_40_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_40_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_40_modules_attn_modules_qkv_parameters_bias_ = (
            L_self_modules_blocks_modules_40_modules_attn_modules_qkv_parameters_bias_
        )
        l_self_modules_blocks_modules_40_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_40_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_40_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_40_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_40_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_40_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_40_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_40_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_40_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_40_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_40_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_40_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_40_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_40_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_40_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_40_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_41_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_41_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_41_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_41_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_41_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_41_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_41_modules_attn_modules_qkv_parameters_bias_ = (
            L_self_modules_blocks_modules_41_modules_attn_modules_qkv_parameters_bias_
        )
        l_self_modules_blocks_modules_41_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_41_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_41_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_41_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_41_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_41_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_41_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_41_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_41_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_41_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_41_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_41_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_41_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_41_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_41_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_41_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_42_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_42_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_42_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_42_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_42_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_42_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_42_modules_attn_modules_qkv_parameters_bias_ = (
            L_self_modules_blocks_modules_42_modules_attn_modules_qkv_parameters_bias_
        )
        l_self_modules_blocks_modules_42_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_42_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_42_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_42_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_42_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_42_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_42_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_42_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_42_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_42_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_42_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_42_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_42_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_42_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_42_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_42_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_43_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_43_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_43_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_43_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_43_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_43_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_43_modules_attn_modules_qkv_parameters_bias_ = (
            L_self_modules_blocks_modules_43_modules_attn_modules_qkv_parameters_bias_
        )
        l_self_modules_blocks_modules_43_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_43_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_43_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_43_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_43_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_43_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_43_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_43_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_43_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_43_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_43_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_43_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_43_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_43_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_43_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_43_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_44_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_44_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_44_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_44_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_44_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_44_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_44_modules_attn_modules_qkv_parameters_bias_ = (
            L_self_modules_blocks_modules_44_modules_attn_modules_qkv_parameters_bias_
        )
        l_self_modules_blocks_modules_44_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_44_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_44_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_44_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_44_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_44_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_44_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_44_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_44_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_44_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_44_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_44_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_44_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_44_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_44_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_44_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_45_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_45_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_45_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_45_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_45_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_45_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_45_modules_attn_modules_qkv_parameters_bias_ = (
            L_self_modules_blocks_modules_45_modules_attn_modules_qkv_parameters_bias_
        )
        l_self_modules_blocks_modules_45_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_45_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_45_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_45_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_45_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_45_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_45_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_45_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_45_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_45_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_45_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_45_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_45_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_45_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_45_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_45_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_46_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_46_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_46_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_46_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_46_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_46_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_46_modules_attn_modules_qkv_parameters_bias_ = (
            L_self_modules_blocks_modules_46_modules_attn_modules_qkv_parameters_bias_
        )
        l_self_modules_blocks_modules_46_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_46_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_46_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_46_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_46_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_46_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_46_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_46_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_46_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_46_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_46_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_46_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_46_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_46_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_46_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_46_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_47_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_47_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_47_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_47_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_47_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_47_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_47_modules_attn_modules_qkv_parameters_bias_ = (
            L_self_modules_blocks_modules_47_modules_attn_modules_qkv_parameters_bias_
        )
        l_self_modules_blocks_modules_47_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_47_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_47_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_47_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_47_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_47_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_47_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_47_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_47_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_47_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_47_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_47_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_47_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_47_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_47_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_47_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_norm_parameters_weight_ = L_self_modules_norm_parameters_weight_
        l_self_modules_norm_parameters_bias_ = L_self_modules_norm_parameters_bias_
        l_self_modules_head_parameters_weight_ = L_self_modules_head_parameters_weight_
        l_self_modules_head_parameters_bias_ = L_self_modules_head_parameters_bias_
        x = torch.conv2d(
            l_x_,
            l_self_modules_patch_embed_modules_proj_parameters_weight_,
            None,
            (14, 14),
            (0, 0),
            (1, 1),
            1,
        )
        l_x_ = l_self_modules_patch_embed_modules_proj_parameters_weight_ = None
        flatten = x.flatten(2)
        x = None
        x_1 = flatten.transpose(1, 2)
        flatten = None
        expand = l_self_parameters_cls_token_.expand(1, -1, -1)
        l_self_parameters_cls_token_ = None
        x_2 = torch.cat([expand, x_1], dim=1)
        expand = x_1 = None
        x_3 = x_2 + l_self_parameters_pos_embed_
        x_2 = l_self_parameters_pos_embed_ = None
        x_4 = torch.nn.functional.dropout(x_3, 0.0, False, False)
        x_3 = None
        x_5 = torch.nn.functional.layer_norm(
            x_4,
            (1664,),
            l_self_modules_norm_pre_parameters_weight_,
            l_self_modules_norm_pre_parameters_bias_,
            1e-05,
        )
        x_4 = (
            l_self_modules_norm_pre_parameters_weight_
        ) = l_self_modules_norm_pre_parameters_bias_ = None
        x_6 = torch.nn.functional.layer_norm(
            x_5,
            (1664,),
            l_self_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_0_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_0_modules_norm1_parameters_bias_
        ) = None
        linear = torch._C._nn.linear(
            x_6,
            l_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_,
        )
        x_6 = (
            l_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape = linear.reshape(1, 257, 3, 16, 104)
        linear = None
        qkv = reshape.permute(2, 0, 3, 1, 4)
        reshape = None
        unbind = qkv.unbind(0)
        qkv = None
        q = unbind[0]
        k = unbind[1]
        v = unbind[2]
        unbind = None
        x_7 = torch._C._nn.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0
        )
        q = k = v = None
        transpose_1 = x_7.transpose(1, 2)
        x_7 = None
        x_8 = transpose_1.reshape(1, 257, 1664)
        transpose_1 = None
        x_9 = torch._C._nn.linear(
            x_8,
            l_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_8 = (
            l_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_10 = torch.nn.functional.dropout(x_9, 0.0, False, False)
        x_9 = None
        x_11 = x_5 + x_10
        x_5 = x_10 = None
        x_12 = torch.nn.functional.layer_norm(
            x_11,
            (1664,),
            l_self_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_0_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_0_modules_norm2_parameters_bias_
        ) = None
        x_13 = torch._C._nn.linear(
            x_12,
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_12 = (
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul = 1.702 * x_13
        sigmoid = torch.sigmoid(mul)
        mul = None
        x_14 = x_13 * sigmoid
        x_13 = sigmoid = None
        x_15 = torch.nn.functional.dropout(x_14, 0.0, False, False)
        x_14 = None
        x_16 = torch._C._nn.linear(
            x_15,
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_15 = (
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_17 = torch.nn.functional.dropout(x_16, 0.0, False, False)
        x_16 = None
        x_18 = x_11 + x_17
        x_11 = x_17 = None
        x_19 = torch.nn.functional.layer_norm(
            x_18,
            (1664,),
            l_self_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_1_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_1_modules_norm1_parameters_bias_
        ) = None
        linear_4 = torch._C._nn.linear(
            x_19,
            l_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_,
        )
        x_19 = (
            l_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_2 = linear_4.reshape(1, 257, 3, 16, 104)
        linear_4 = None
        qkv_1 = reshape_2.permute(2, 0, 3, 1, 4)
        reshape_2 = None
        unbind_1 = qkv_1.unbind(0)
        qkv_1 = None
        q_1 = unbind_1[0]
        k_1 = unbind_1[1]
        v_1 = unbind_1[2]
        unbind_1 = None
        x_20 = torch._C._nn.scaled_dot_product_attention(
            q_1, k_1, v_1, attn_mask=None, dropout_p=0.0
        )
        q_1 = k_1 = v_1 = None
        transpose_2 = x_20.transpose(1, 2)
        x_20 = None
        x_21 = transpose_2.reshape(1, 257, 1664)
        transpose_2 = None
        x_22 = torch._C._nn.linear(
            x_21,
            l_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_21 = (
            l_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_23 = torch.nn.functional.dropout(x_22, 0.0, False, False)
        x_22 = None
        x_24 = x_18 + x_23
        x_18 = x_23 = None
        x_25 = torch.nn.functional.layer_norm(
            x_24,
            (1664,),
            l_self_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_1_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_1_modules_norm2_parameters_bias_
        ) = None
        x_26 = torch._C._nn.linear(
            x_25,
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_25 = (
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_2 = 1.702 * x_26
        sigmoid_1 = torch.sigmoid(mul_2)
        mul_2 = None
        x_27 = x_26 * sigmoid_1
        x_26 = sigmoid_1 = None
        x_28 = torch.nn.functional.dropout(x_27, 0.0, False, False)
        x_27 = None
        x_29 = torch._C._nn.linear(
            x_28,
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_28 = (
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_30 = torch.nn.functional.dropout(x_29, 0.0, False, False)
        x_29 = None
        x_31 = x_24 + x_30
        x_24 = x_30 = None
        x_32 = torch.nn.functional.layer_norm(
            x_31,
            (1664,),
            l_self_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_2_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_2_modules_norm1_parameters_bias_
        ) = None
        linear_8 = torch._C._nn.linear(
            x_32,
            l_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_,
        )
        x_32 = (
            l_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_4 = linear_8.reshape(1, 257, 3, 16, 104)
        linear_8 = None
        qkv_2 = reshape_4.permute(2, 0, 3, 1, 4)
        reshape_4 = None
        unbind_2 = qkv_2.unbind(0)
        qkv_2 = None
        q_2 = unbind_2[0]
        k_2 = unbind_2[1]
        v_2 = unbind_2[2]
        unbind_2 = None
        x_33 = torch._C._nn.scaled_dot_product_attention(
            q_2, k_2, v_2, attn_mask=None, dropout_p=0.0
        )
        q_2 = k_2 = v_2 = None
        transpose_3 = x_33.transpose(1, 2)
        x_33 = None
        x_34 = transpose_3.reshape(1, 257, 1664)
        transpose_3 = None
        x_35 = torch._C._nn.linear(
            x_34,
            l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_,
        )
        x_34 = (
            l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_36 = torch.nn.functional.dropout(x_35, 0.0, False, False)
        x_35 = None
        x_37 = x_31 + x_36
        x_31 = x_36 = None
        x_38 = torch.nn.functional.layer_norm(
            x_37,
            (1664,),
            l_self_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_2_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_2_modules_norm2_parameters_bias_
        ) = None
        x_39 = torch._C._nn.linear(
            x_38,
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_38 = (
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_4 = 1.702 * x_39
        sigmoid_2 = torch.sigmoid(mul_4)
        mul_4 = None
        x_40 = x_39 * sigmoid_2
        x_39 = sigmoid_2 = None
        x_41 = torch.nn.functional.dropout(x_40, 0.0, False, False)
        x_40 = None
        x_42 = torch._C._nn.linear(
            x_41,
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_41 = (
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_43 = torch.nn.functional.dropout(x_42, 0.0, False, False)
        x_42 = None
        x_44 = x_37 + x_43
        x_37 = x_43 = None
        x_45 = torch.nn.functional.layer_norm(
            x_44,
            (1664,),
            l_self_modules_blocks_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_3_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_3_modules_norm1_parameters_bias_
        ) = None
        linear_12 = torch._C._nn.linear(
            x_45,
            l_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_bias_,
        )
        x_45 = (
            l_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_6 = linear_12.reshape(1, 257, 3, 16, 104)
        linear_12 = None
        qkv_3 = reshape_6.permute(2, 0, 3, 1, 4)
        reshape_6 = None
        unbind_3 = qkv_3.unbind(0)
        qkv_3 = None
        q_3 = unbind_3[0]
        k_3 = unbind_3[1]
        v_3 = unbind_3[2]
        unbind_3 = None
        x_46 = torch._C._nn.scaled_dot_product_attention(
            q_3, k_3, v_3, attn_mask=None, dropout_p=0.0
        )
        q_3 = k_3 = v_3 = None
        transpose_4 = x_46.transpose(1, 2)
        x_46 = None
        x_47 = transpose_4.reshape(1, 257, 1664)
        transpose_4 = None
        x_48 = torch._C._nn.linear(
            x_47,
            l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_,
        )
        x_47 = (
            l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_49 = torch.nn.functional.dropout(x_48, 0.0, False, False)
        x_48 = None
        x_50 = x_44 + x_49
        x_44 = x_49 = None
        x_51 = torch.nn.functional.layer_norm(
            x_50,
            (1664,),
            l_self_modules_blocks_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_3_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_3_modules_norm2_parameters_bias_
        ) = None
        x_52 = torch._C._nn.linear(
            x_51,
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_51 = (
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_6 = 1.702 * x_52
        sigmoid_3 = torch.sigmoid(mul_6)
        mul_6 = None
        x_53 = x_52 * sigmoid_3
        x_52 = sigmoid_3 = None
        x_54 = torch.nn.functional.dropout(x_53, 0.0, False, False)
        x_53 = None
        x_55 = torch._C._nn.linear(
            x_54,
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_54 = (
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_56 = torch.nn.functional.dropout(x_55, 0.0, False, False)
        x_55 = None
        x_57 = x_50 + x_56
        x_50 = x_56 = None
        x_58 = torch.nn.functional.layer_norm(
            x_57,
            (1664,),
            l_self_modules_blocks_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_4_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_4_modules_norm1_parameters_bias_
        ) = None
        linear_16 = torch._C._nn.linear(
            x_58,
            l_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_bias_,
        )
        x_58 = (
            l_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_8 = linear_16.reshape(1, 257, 3, 16, 104)
        linear_16 = None
        qkv_4 = reshape_8.permute(2, 0, 3, 1, 4)
        reshape_8 = None
        unbind_4 = qkv_4.unbind(0)
        qkv_4 = None
        q_4 = unbind_4[0]
        k_4 = unbind_4[1]
        v_4 = unbind_4[2]
        unbind_4 = None
        x_59 = torch._C._nn.scaled_dot_product_attention(
            q_4, k_4, v_4, attn_mask=None, dropout_p=0.0
        )
        q_4 = k_4 = v_4 = None
        transpose_5 = x_59.transpose(1, 2)
        x_59 = None
        x_60 = transpose_5.reshape(1, 257, 1664)
        transpose_5 = None
        x_61 = torch._C._nn.linear(
            x_60,
            l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_,
        )
        x_60 = (
            l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_62 = torch.nn.functional.dropout(x_61, 0.0, False, False)
        x_61 = None
        x_63 = x_57 + x_62
        x_57 = x_62 = None
        x_64 = torch.nn.functional.layer_norm(
            x_63,
            (1664,),
            l_self_modules_blocks_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_4_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_4_modules_norm2_parameters_bias_
        ) = None
        x_65 = torch._C._nn.linear(
            x_64,
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_64 = (
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_8 = 1.702 * x_65
        sigmoid_4 = torch.sigmoid(mul_8)
        mul_8 = None
        x_66 = x_65 * sigmoid_4
        x_65 = sigmoid_4 = None
        x_67 = torch.nn.functional.dropout(x_66, 0.0, False, False)
        x_66 = None
        x_68 = torch._C._nn.linear(
            x_67,
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_67 = (
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_69 = torch.nn.functional.dropout(x_68, 0.0, False, False)
        x_68 = None
        x_70 = x_63 + x_69
        x_63 = x_69 = None
        x_71 = torch.nn.functional.layer_norm(
            x_70,
            (1664,),
            l_self_modules_blocks_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_5_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_5_modules_norm1_parameters_bias_
        ) = None
        linear_20 = torch._C._nn.linear(
            x_71,
            l_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_bias_,
        )
        x_71 = (
            l_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_10 = linear_20.reshape(1, 257, 3, 16, 104)
        linear_20 = None
        qkv_5 = reshape_10.permute(2, 0, 3, 1, 4)
        reshape_10 = None
        unbind_5 = qkv_5.unbind(0)
        qkv_5 = None
        q_5 = unbind_5[0]
        k_5 = unbind_5[1]
        v_5 = unbind_5[2]
        unbind_5 = None
        x_72 = torch._C._nn.scaled_dot_product_attention(
            q_5, k_5, v_5, attn_mask=None, dropout_p=0.0
        )
        q_5 = k_5 = v_5 = None
        transpose_6 = x_72.transpose(1, 2)
        x_72 = None
        x_73 = transpose_6.reshape(1, 257, 1664)
        transpose_6 = None
        x_74 = torch._C._nn.linear(
            x_73,
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_,
        )
        x_73 = (
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_75 = torch.nn.functional.dropout(x_74, 0.0, False, False)
        x_74 = None
        x_76 = x_70 + x_75
        x_70 = x_75 = None
        x_77 = torch.nn.functional.layer_norm(
            x_76,
            (1664,),
            l_self_modules_blocks_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_5_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_5_modules_norm2_parameters_bias_
        ) = None
        x_78 = torch._C._nn.linear(
            x_77,
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_77 = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_10 = 1.702 * x_78
        sigmoid_5 = torch.sigmoid(mul_10)
        mul_10 = None
        x_79 = x_78 * sigmoid_5
        x_78 = sigmoid_5 = None
        x_80 = torch.nn.functional.dropout(x_79, 0.0, False, False)
        x_79 = None
        x_81 = torch._C._nn.linear(
            x_80,
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_80 = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_82 = torch.nn.functional.dropout(x_81, 0.0, False, False)
        x_81 = None
        x_83 = x_76 + x_82
        x_76 = x_82 = None
        x_84 = torch.nn.functional.layer_norm(
            x_83,
            (1664,),
            l_self_modules_blocks_modules_6_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_6_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_6_modules_norm1_parameters_bias_
        ) = None
        linear_24 = torch._C._nn.linear(
            x_84,
            l_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_bias_,
        )
        x_84 = (
            l_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_12 = linear_24.reshape(1, 257, 3, 16, 104)
        linear_24 = None
        qkv_6 = reshape_12.permute(2, 0, 3, 1, 4)
        reshape_12 = None
        unbind_6 = qkv_6.unbind(0)
        qkv_6 = None
        q_6 = unbind_6[0]
        k_6 = unbind_6[1]
        v_6 = unbind_6[2]
        unbind_6 = None
        x_85 = torch._C._nn.scaled_dot_product_attention(
            q_6, k_6, v_6, attn_mask=None, dropout_p=0.0
        )
        q_6 = k_6 = v_6 = None
        transpose_7 = x_85.transpose(1, 2)
        x_85 = None
        x_86 = transpose_7.reshape(1, 257, 1664)
        transpose_7 = None
        x_87 = torch._C._nn.linear(
            x_86,
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_,
        )
        x_86 = (
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_88 = torch.nn.functional.dropout(x_87, 0.0, False, False)
        x_87 = None
        x_89 = x_83 + x_88
        x_83 = x_88 = None
        x_90 = torch.nn.functional.layer_norm(
            x_89,
            (1664,),
            l_self_modules_blocks_modules_6_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_6_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_6_modules_norm2_parameters_bias_
        ) = None
        x_91 = torch._C._nn.linear(
            x_90,
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_90 = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_12 = 1.702 * x_91
        sigmoid_6 = torch.sigmoid(mul_12)
        mul_12 = None
        x_92 = x_91 * sigmoid_6
        x_91 = sigmoid_6 = None
        x_93 = torch.nn.functional.dropout(x_92, 0.0, False, False)
        x_92 = None
        x_94 = torch._C._nn.linear(
            x_93,
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_93 = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_95 = torch.nn.functional.dropout(x_94, 0.0, False, False)
        x_94 = None
        x_96 = x_89 + x_95
        x_89 = x_95 = None
        x_97 = torch.nn.functional.layer_norm(
            x_96,
            (1664,),
            l_self_modules_blocks_modules_7_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_7_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_7_modules_norm1_parameters_bias_
        ) = None
        linear_28 = torch._C._nn.linear(
            x_97,
            l_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_bias_,
        )
        x_97 = (
            l_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_14 = linear_28.reshape(1, 257, 3, 16, 104)
        linear_28 = None
        qkv_7 = reshape_14.permute(2, 0, 3, 1, 4)
        reshape_14 = None
        unbind_7 = qkv_7.unbind(0)
        qkv_7 = None
        q_7 = unbind_7[0]
        k_7 = unbind_7[1]
        v_7 = unbind_7[2]
        unbind_7 = None
        x_98 = torch._C._nn.scaled_dot_product_attention(
            q_7, k_7, v_7, attn_mask=None, dropout_p=0.0
        )
        q_7 = k_7 = v_7 = None
        transpose_8 = x_98.transpose(1, 2)
        x_98 = None
        x_99 = transpose_8.reshape(1, 257, 1664)
        transpose_8 = None
        x_100 = torch._C._nn.linear(
            x_99,
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_,
        )
        x_99 = (
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_101 = torch.nn.functional.dropout(x_100, 0.0, False, False)
        x_100 = None
        x_102 = x_96 + x_101
        x_96 = x_101 = None
        x_103 = torch.nn.functional.layer_norm(
            x_102,
            (1664,),
            l_self_modules_blocks_modules_7_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_7_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_7_modules_norm2_parameters_bias_
        ) = None
        x_104 = torch._C._nn.linear(
            x_103,
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_103 = (
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_14 = 1.702 * x_104
        sigmoid_7 = torch.sigmoid(mul_14)
        mul_14 = None
        x_105 = x_104 * sigmoid_7
        x_104 = sigmoid_7 = None
        x_106 = torch.nn.functional.dropout(x_105, 0.0, False, False)
        x_105 = None
        x_107 = torch._C._nn.linear(
            x_106,
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_106 = (
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_108 = torch.nn.functional.dropout(x_107, 0.0, False, False)
        x_107 = None
        x_109 = x_102 + x_108
        x_102 = x_108 = None
        x_110 = torch.nn.functional.layer_norm(
            x_109,
            (1664,),
            l_self_modules_blocks_modules_8_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_8_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_8_modules_norm1_parameters_bias_
        ) = None
        linear_32 = torch._C._nn.linear(
            x_110,
            l_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_bias_,
        )
        x_110 = (
            l_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_16 = linear_32.reshape(1, 257, 3, 16, 104)
        linear_32 = None
        qkv_8 = reshape_16.permute(2, 0, 3, 1, 4)
        reshape_16 = None
        unbind_8 = qkv_8.unbind(0)
        qkv_8 = None
        q_8 = unbind_8[0]
        k_8 = unbind_8[1]
        v_8 = unbind_8[2]
        unbind_8 = None
        x_111 = torch._C._nn.scaled_dot_product_attention(
            q_8, k_8, v_8, attn_mask=None, dropout_p=0.0
        )
        q_8 = k_8 = v_8 = None
        transpose_9 = x_111.transpose(1, 2)
        x_111 = None
        x_112 = transpose_9.reshape(1, 257, 1664)
        transpose_9 = None
        x_113 = torch._C._nn.linear(
            x_112,
            l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_,
        )
        x_112 = (
            l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_114 = torch.nn.functional.dropout(x_113, 0.0, False, False)
        x_113 = None
        x_115 = x_109 + x_114
        x_109 = x_114 = None
        x_116 = torch.nn.functional.layer_norm(
            x_115,
            (1664,),
            l_self_modules_blocks_modules_8_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_8_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_8_modules_norm2_parameters_bias_
        ) = None
        x_117 = torch._C._nn.linear(
            x_116,
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_116 = (
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_16 = 1.702 * x_117
        sigmoid_8 = torch.sigmoid(mul_16)
        mul_16 = None
        x_118 = x_117 * sigmoid_8
        x_117 = sigmoid_8 = None
        x_119 = torch.nn.functional.dropout(x_118, 0.0, False, False)
        x_118 = None
        x_120 = torch._C._nn.linear(
            x_119,
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_119 = (
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_121 = torch.nn.functional.dropout(x_120, 0.0, False, False)
        x_120 = None
        x_122 = x_115 + x_121
        x_115 = x_121 = None
        x_123 = torch.nn.functional.layer_norm(
            x_122,
            (1664,),
            l_self_modules_blocks_modules_9_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_9_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_9_modules_norm1_parameters_bias_
        ) = None
        linear_36 = torch._C._nn.linear(
            x_123,
            l_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_bias_,
        )
        x_123 = (
            l_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_18 = linear_36.reshape(1, 257, 3, 16, 104)
        linear_36 = None
        qkv_9 = reshape_18.permute(2, 0, 3, 1, 4)
        reshape_18 = None
        unbind_9 = qkv_9.unbind(0)
        qkv_9 = None
        q_9 = unbind_9[0]
        k_9 = unbind_9[1]
        v_9 = unbind_9[2]
        unbind_9 = None
        x_124 = torch._C._nn.scaled_dot_product_attention(
            q_9, k_9, v_9, attn_mask=None, dropout_p=0.0
        )
        q_9 = k_9 = v_9 = None
        transpose_10 = x_124.transpose(1, 2)
        x_124 = None
        x_125 = transpose_10.reshape(1, 257, 1664)
        transpose_10 = None
        x_126 = torch._C._nn.linear(
            x_125,
            l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_,
        )
        x_125 = (
            l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_127 = torch.nn.functional.dropout(x_126, 0.0, False, False)
        x_126 = None
        x_128 = x_122 + x_127
        x_122 = x_127 = None
        x_129 = torch.nn.functional.layer_norm(
            x_128,
            (1664,),
            l_self_modules_blocks_modules_9_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_9_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_9_modules_norm2_parameters_bias_
        ) = None
        x_130 = torch._C._nn.linear(
            x_129,
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_129 = (
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_18 = 1.702 * x_130
        sigmoid_9 = torch.sigmoid(mul_18)
        mul_18 = None
        x_131 = x_130 * sigmoid_9
        x_130 = sigmoid_9 = None
        x_132 = torch.nn.functional.dropout(x_131, 0.0, False, False)
        x_131 = None
        x_133 = torch._C._nn.linear(
            x_132,
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_132 = (
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_134 = torch.nn.functional.dropout(x_133, 0.0, False, False)
        x_133 = None
        x_135 = x_128 + x_134
        x_128 = x_134 = None
        x_136 = torch.nn.functional.layer_norm(
            x_135,
            (1664,),
            l_self_modules_blocks_modules_10_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_10_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_10_modules_norm1_parameters_bias_
        ) = None
        linear_40 = torch._C._nn.linear(
            x_136,
            l_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_bias_,
        )
        x_136 = (
            l_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_20 = linear_40.reshape(1, 257, 3, 16, 104)
        linear_40 = None
        qkv_10 = reshape_20.permute(2, 0, 3, 1, 4)
        reshape_20 = None
        unbind_10 = qkv_10.unbind(0)
        qkv_10 = None
        q_10 = unbind_10[0]
        k_10 = unbind_10[1]
        v_10 = unbind_10[2]
        unbind_10 = None
        x_137 = torch._C._nn.scaled_dot_product_attention(
            q_10, k_10, v_10, attn_mask=None, dropout_p=0.0
        )
        q_10 = k_10 = v_10 = None
        transpose_11 = x_137.transpose(1, 2)
        x_137 = None
        x_138 = transpose_11.reshape(1, 257, 1664)
        transpose_11 = None
        x_139 = torch._C._nn.linear(
            x_138,
            l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_,
        )
        x_138 = l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_140 = torch.nn.functional.dropout(x_139, 0.0, False, False)
        x_139 = None
        x_141 = x_135 + x_140
        x_135 = x_140 = None
        x_142 = torch.nn.functional.layer_norm(
            x_141,
            (1664,),
            l_self_modules_blocks_modules_10_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_10_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_10_modules_norm2_parameters_bias_
        ) = None
        x_143 = torch._C._nn.linear(
            x_142,
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_142 = (
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_20 = 1.702 * x_143
        sigmoid_10 = torch.sigmoid(mul_20)
        mul_20 = None
        x_144 = x_143 * sigmoid_10
        x_143 = sigmoid_10 = None
        x_145 = torch.nn.functional.dropout(x_144, 0.0, False, False)
        x_144 = None
        x_146 = torch._C._nn.linear(
            x_145,
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_145 = (
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_147 = torch.nn.functional.dropout(x_146, 0.0, False, False)
        x_146 = None
        x_148 = x_141 + x_147
        x_141 = x_147 = None
        x_149 = torch.nn.functional.layer_norm(
            x_148,
            (1664,),
            l_self_modules_blocks_modules_11_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_11_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_11_modules_norm1_parameters_bias_
        ) = None
        linear_44 = torch._C._nn.linear(
            x_149,
            l_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_bias_,
        )
        x_149 = (
            l_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_22 = linear_44.reshape(1, 257, 3, 16, 104)
        linear_44 = None
        qkv_11 = reshape_22.permute(2, 0, 3, 1, 4)
        reshape_22 = None
        unbind_11 = qkv_11.unbind(0)
        qkv_11 = None
        q_11 = unbind_11[0]
        k_11 = unbind_11[1]
        v_11 = unbind_11[2]
        unbind_11 = None
        x_150 = torch._C._nn.scaled_dot_product_attention(
            q_11, k_11, v_11, attn_mask=None, dropout_p=0.0
        )
        q_11 = k_11 = v_11 = None
        transpose_12 = x_150.transpose(1, 2)
        x_150 = None
        x_151 = transpose_12.reshape(1, 257, 1664)
        transpose_12 = None
        x_152 = torch._C._nn.linear(
            x_151,
            l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_,
        )
        x_151 = l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_153 = torch.nn.functional.dropout(x_152, 0.0, False, False)
        x_152 = None
        x_154 = x_148 + x_153
        x_148 = x_153 = None
        x_155 = torch.nn.functional.layer_norm(
            x_154,
            (1664,),
            l_self_modules_blocks_modules_11_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_11_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_11_modules_norm2_parameters_bias_
        ) = None
        x_156 = torch._C._nn.linear(
            x_155,
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_155 = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_22 = 1.702 * x_156
        sigmoid_11 = torch.sigmoid(mul_22)
        mul_22 = None
        x_157 = x_156 * sigmoid_11
        x_156 = sigmoid_11 = None
        x_158 = torch.nn.functional.dropout(x_157, 0.0, False, False)
        x_157 = None
        x_159 = torch._C._nn.linear(
            x_158,
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_158 = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_160 = torch.nn.functional.dropout(x_159, 0.0, False, False)
        x_159 = None
        x_161 = x_154 + x_160
        x_154 = x_160 = None
        x_162 = torch.nn.functional.layer_norm(
            x_161,
            (1664,),
            l_self_modules_blocks_modules_12_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_12_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_12_modules_norm1_parameters_bias_
        ) = None
        linear_48 = torch._C._nn.linear(
            x_162,
            l_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_bias_,
        )
        x_162 = (
            l_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_24 = linear_48.reshape(1, 257, 3, 16, 104)
        linear_48 = None
        qkv_12 = reshape_24.permute(2, 0, 3, 1, 4)
        reshape_24 = None
        unbind_12 = qkv_12.unbind(0)
        qkv_12 = None
        q_12 = unbind_12[0]
        k_12 = unbind_12[1]
        v_12 = unbind_12[2]
        unbind_12 = None
        x_163 = torch._C._nn.scaled_dot_product_attention(
            q_12, k_12, v_12, attn_mask=None, dropout_p=0.0
        )
        q_12 = k_12 = v_12 = None
        transpose_13 = x_163.transpose(1, 2)
        x_163 = None
        x_164 = transpose_13.reshape(1, 257, 1664)
        transpose_13 = None
        x_165 = torch._C._nn.linear(
            x_164,
            l_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_,
        )
        x_164 = l_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_166 = torch.nn.functional.dropout(x_165, 0.0, False, False)
        x_165 = None
        x_167 = x_161 + x_166
        x_161 = x_166 = None
        x_168 = torch.nn.functional.layer_norm(
            x_167,
            (1664,),
            l_self_modules_blocks_modules_12_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_12_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_12_modules_norm2_parameters_bias_
        ) = None
        x_169 = torch._C._nn.linear(
            x_168,
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_168 = (
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_24 = 1.702 * x_169
        sigmoid_12 = torch.sigmoid(mul_24)
        mul_24 = None
        x_170 = x_169 * sigmoid_12
        x_169 = sigmoid_12 = None
        x_171 = torch.nn.functional.dropout(x_170, 0.0, False, False)
        x_170 = None
        x_172 = torch._C._nn.linear(
            x_171,
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_171 = (
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_173 = torch.nn.functional.dropout(x_172, 0.0, False, False)
        x_172 = None
        x_174 = x_167 + x_173
        x_167 = x_173 = None
        x_175 = torch.nn.functional.layer_norm(
            x_174,
            (1664,),
            l_self_modules_blocks_modules_13_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_13_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_13_modules_norm1_parameters_bias_
        ) = None
        linear_52 = torch._C._nn.linear(
            x_175,
            l_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_bias_,
        )
        x_175 = (
            l_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_26 = linear_52.reshape(1, 257, 3, 16, 104)
        linear_52 = None
        qkv_13 = reshape_26.permute(2, 0, 3, 1, 4)
        reshape_26 = None
        unbind_13 = qkv_13.unbind(0)
        qkv_13 = None
        q_13 = unbind_13[0]
        k_13 = unbind_13[1]
        v_13 = unbind_13[2]
        unbind_13 = None
        x_176 = torch._C._nn.scaled_dot_product_attention(
            q_13, k_13, v_13, attn_mask=None, dropout_p=0.0
        )
        q_13 = k_13 = v_13 = None
        transpose_14 = x_176.transpose(1, 2)
        x_176 = None
        x_177 = transpose_14.reshape(1, 257, 1664)
        transpose_14 = None
        x_178 = torch._C._nn.linear(
            x_177,
            l_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_,
        )
        x_177 = l_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_179 = torch.nn.functional.dropout(x_178, 0.0, False, False)
        x_178 = None
        x_180 = x_174 + x_179
        x_174 = x_179 = None
        x_181 = torch.nn.functional.layer_norm(
            x_180,
            (1664,),
            l_self_modules_blocks_modules_13_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_13_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_13_modules_norm2_parameters_bias_
        ) = None
        x_182 = torch._C._nn.linear(
            x_181,
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_181 = (
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_26 = 1.702 * x_182
        sigmoid_13 = torch.sigmoid(mul_26)
        mul_26 = None
        x_183 = x_182 * sigmoid_13
        x_182 = sigmoid_13 = None
        x_184 = torch.nn.functional.dropout(x_183, 0.0, False, False)
        x_183 = None
        x_185 = torch._C._nn.linear(
            x_184,
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_184 = (
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_186 = torch.nn.functional.dropout(x_185, 0.0, False, False)
        x_185 = None
        x_187 = x_180 + x_186
        x_180 = x_186 = None
        x_188 = torch.nn.functional.layer_norm(
            x_187,
            (1664,),
            l_self_modules_blocks_modules_14_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_14_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_14_modules_norm1_parameters_bias_
        ) = None
        linear_56 = torch._C._nn.linear(
            x_188,
            l_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_bias_,
        )
        x_188 = (
            l_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_28 = linear_56.reshape(1, 257, 3, 16, 104)
        linear_56 = None
        qkv_14 = reshape_28.permute(2, 0, 3, 1, 4)
        reshape_28 = None
        unbind_14 = qkv_14.unbind(0)
        qkv_14 = None
        q_14 = unbind_14[0]
        k_14 = unbind_14[1]
        v_14 = unbind_14[2]
        unbind_14 = None
        x_189 = torch._C._nn.scaled_dot_product_attention(
            q_14, k_14, v_14, attn_mask=None, dropout_p=0.0
        )
        q_14 = k_14 = v_14 = None
        transpose_15 = x_189.transpose(1, 2)
        x_189 = None
        x_190 = transpose_15.reshape(1, 257, 1664)
        transpose_15 = None
        x_191 = torch._C._nn.linear(
            x_190,
            l_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_bias_,
        )
        x_190 = l_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_192 = torch.nn.functional.dropout(x_191, 0.0, False, False)
        x_191 = None
        x_193 = x_187 + x_192
        x_187 = x_192 = None
        x_194 = torch.nn.functional.layer_norm(
            x_193,
            (1664,),
            l_self_modules_blocks_modules_14_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_14_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_14_modules_norm2_parameters_bias_
        ) = None
        x_195 = torch._C._nn.linear(
            x_194,
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_194 = (
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_28 = 1.702 * x_195
        sigmoid_14 = torch.sigmoid(mul_28)
        mul_28 = None
        x_196 = x_195 * sigmoid_14
        x_195 = sigmoid_14 = None
        x_197 = torch.nn.functional.dropout(x_196, 0.0, False, False)
        x_196 = None
        x_198 = torch._C._nn.linear(
            x_197,
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_197 = (
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_199 = torch.nn.functional.dropout(x_198, 0.0, False, False)
        x_198 = None
        x_200 = x_193 + x_199
        x_193 = x_199 = None
        x_201 = torch.nn.functional.layer_norm(
            x_200,
            (1664,),
            l_self_modules_blocks_modules_15_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_15_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_15_modules_norm1_parameters_bias_
        ) = None
        linear_60 = torch._C._nn.linear(
            x_201,
            l_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_bias_,
        )
        x_201 = (
            l_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_30 = linear_60.reshape(1, 257, 3, 16, 104)
        linear_60 = None
        qkv_15 = reshape_30.permute(2, 0, 3, 1, 4)
        reshape_30 = None
        unbind_15 = qkv_15.unbind(0)
        qkv_15 = None
        q_15 = unbind_15[0]
        k_15 = unbind_15[1]
        v_15 = unbind_15[2]
        unbind_15 = None
        x_202 = torch._C._nn.scaled_dot_product_attention(
            q_15, k_15, v_15, attn_mask=None, dropout_p=0.0
        )
        q_15 = k_15 = v_15 = None
        transpose_16 = x_202.transpose(1, 2)
        x_202 = None
        x_203 = transpose_16.reshape(1, 257, 1664)
        transpose_16 = None
        x_204 = torch._C._nn.linear(
            x_203,
            l_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_bias_,
        )
        x_203 = l_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_205 = torch.nn.functional.dropout(x_204, 0.0, False, False)
        x_204 = None
        x_206 = x_200 + x_205
        x_200 = x_205 = None
        x_207 = torch.nn.functional.layer_norm(
            x_206,
            (1664,),
            l_self_modules_blocks_modules_15_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_15_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_15_modules_norm2_parameters_bias_
        ) = None
        x_208 = torch._C._nn.linear(
            x_207,
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_207 = (
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_30 = 1.702 * x_208
        sigmoid_15 = torch.sigmoid(mul_30)
        mul_30 = None
        x_209 = x_208 * sigmoid_15
        x_208 = sigmoid_15 = None
        x_210 = torch.nn.functional.dropout(x_209, 0.0, False, False)
        x_209 = None
        x_211 = torch._C._nn.linear(
            x_210,
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_210 = (
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_212 = torch.nn.functional.dropout(x_211, 0.0, False, False)
        x_211 = None
        x_213 = x_206 + x_212
        x_206 = x_212 = None
        x_214 = torch.nn.functional.layer_norm(
            x_213,
            (1664,),
            l_self_modules_blocks_modules_16_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_16_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_16_modules_norm1_parameters_bias_
        ) = None
        linear_64 = torch._C._nn.linear(
            x_214,
            l_self_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_bias_,
        )
        x_214 = (
            l_self_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_32 = linear_64.reshape(1, 257, 3, 16, 104)
        linear_64 = None
        qkv_16 = reshape_32.permute(2, 0, 3, 1, 4)
        reshape_32 = None
        unbind_16 = qkv_16.unbind(0)
        qkv_16 = None
        q_16 = unbind_16[0]
        k_16 = unbind_16[1]
        v_16 = unbind_16[2]
        unbind_16 = None
        x_215 = torch._C._nn.scaled_dot_product_attention(
            q_16, k_16, v_16, attn_mask=None, dropout_p=0.0
        )
        q_16 = k_16 = v_16 = None
        transpose_17 = x_215.transpose(1, 2)
        x_215 = None
        x_216 = transpose_17.reshape(1, 257, 1664)
        transpose_17 = None
        x_217 = torch._C._nn.linear(
            x_216,
            l_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_bias_,
        )
        x_216 = l_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_218 = torch.nn.functional.dropout(x_217, 0.0, False, False)
        x_217 = None
        x_219 = x_213 + x_218
        x_213 = x_218 = None
        x_220 = torch.nn.functional.layer_norm(
            x_219,
            (1664,),
            l_self_modules_blocks_modules_16_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_16_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_16_modules_norm2_parameters_bias_
        ) = None
        x_221 = torch._C._nn.linear(
            x_220,
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_220 = (
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_32 = 1.702 * x_221
        sigmoid_16 = torch.sigmoid(mul_32)
        mul_32 = None
        x_222 = x_221 * sigmoid_16
        x_221 = sigmoid_16 = None
        x_223 = torch.nn.functional.dropout(x_222, 0.0, False, False)
        x_222 = None
        x_224 = torch._C._nn.linear(
            x_223,
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_223 = (
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_225 = torch.nn.functional.dropout(x_224, 0.0, False, False)
        x_224 = None
        x_226 = x_219 + x_225
        x_219 = x_225 = None
        x_227 = torch.nn.functional.layer_norm(
            x_226,
            (1664,),
            l_self_modules_blocks_modules_17_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_17_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_17_modules_norm1_parameters_bias_
        ) = None
        linear_68 = torch._C._nn.linear(
            x_227,
            l_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_bias_,
        )
        x_227 = (
            l_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_34 = linear_68.reshape(1, 257, 3, 16, 104)
        linear_68 = None
        qkv_17 = reshape_34.permute(2, 0, 3, 1, 4)
        reshape_34 = None
        unbind_17 = qkv_17.unbind(0)
        qkv_17 = None
        q_17 = unbind_17[0]
        k_17 = unbind_17[1]
        v_17 = unbind_17[2]
        unbind_17 = None
        x_228 = torch._C._nn.scaled_dot_product_attention(
            q_17, k_17, v_17, attn_mask=None, dropout_p=0.0
        )
        q_17 = k_17 = v_17 = None
        transpose_18 = x_228.transpose(1, 2)
        x_228 = None
        x_229 = transpose_18.reshape(1, 257, 1664)
        transpose_18 = None
        x_230 = torch._C._nn.linear(
            x_229,
            l_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_bias_,
        )
        x_229 = l_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_231 = torch.nn.functional.dropout(x_230, 0.0, False, False)
        x_230 = None
        x_232 = x_226 + x_231
        x_226 = x_231 = None
        x_233 = torch.nn.functional.layer_norm(
            x_232,
            (1664,),
            l_self_modules_blocks_modules_17_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_17_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_17_modules_norm2_parameters_bias_
        ) = None
        x_234 = torch._C._nn.linear(
            x_233,
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_233 = (
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_34 = 1.702 * x_234
        sigmoid_17 = torch.sigmoid(mul_34)
        mul_34 = None
        x_235 = x_234 * sigmoid_17
        x_234 = sigmoid_17 = None
        x_236 = torch.nn.functional.dropout(x_235, 0.0, False, False)
        x_235 = None
        x_237 = torch._C._nn.linear(
            x_236,
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_236 = (
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_238 = torch.nn.functional.dropout(x_237, 0.0, False, False)
        x_237 = None
        x_239 = x_232 + x_238
        x_232 = x_238 = None
        x_240 = torch.nn.functional.layer_norm(
            x_239,
            (1664,),
            l_self_modules_blocks_modules_18_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_18_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_18_modules_norm1_parameters_bias_
        ) = None
        linear_72 = torch._C._nn.linear(
            x_240,
            l_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_bias_,
        )
        x_240 = (
            l_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_36 = linear_72.reshape(1, 257, 3, 16, 104)
        linear_72 = None
        qkv_18 = reshape_36.permute(2, 0, 3, 1, 4)
        reshape_36 = None
        unbind_18 = qkv_18.unbind(0)
        qkv_18 = None
        q_18 = unbind_18[0]
        k_18 = unbind_18[1]
        v_18 = unbind_18[2]
        unbind_18 = None
        x_241 = torch._C._nn.scaled_dot_product_attention(
            q_18, k_18, v_18, attn_mask=None, dropout_p=0.0
        )
        q_18 = k_18 = v_18 = None
        transpose_19 = x_241.transpose(1, 2)
        x_241 = None
        x_242 = transpose_19.reshape(1, 257, 1664)
        transpose_19 = None
        x_243 = torch._C._nn.linear(
            x_242,
            l_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_bias_,
        )
        x_242 = l_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_244 = torch.nn.functional.dropout(x_243, 0.0, False, False)
        x_243 = None
        x_245 = x_239 + x_244
        x_239 = x_244 = None
        x_246 = torch.nn.functional.layer_norm(
            x_245,
            (1664,),
            l_self_modules_blocks_modules_18_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_18_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_18_modules_norm2_parameters_bias_
        ) = None
        x_247 = torch._C._nn.linear(
            x_246,
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_246 = (
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_36 = 1.702 * x_247
        sigmoid_18 = torch.sigmoid(mul_36)
        mul_36 = None
        x_248 = x_247 * sigmoid_18
        x_247 = sigmoid_18 = None
        x_249 = torch.nn.functional.dropout(x_248, 0.0, False, False)
        x_248 = None
        x_250 = torch._C._nn.linear(
            x_249,
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_249 = (
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_251 = torch.nn.functional.dropout(x_250, 0.0, False, False)
        x_250 = None
        x_252 = x_245 + x_251
        x_245 = x_251 = None
        x_253 = torch.nn.functional.layer_norm(
            x_252,
            (1664,),
            l_self_modules_blocks_modules_19_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_19_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_19_modules_norm1_parameters_bias_
        ) = None
        linear_76 = torch._C._nn.linear(
            x_253,
            l_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_bias_,
        )
        x_253 = (
            l_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_38 = linear_76.reshape(1, 257, 3, 16, 104)
        linear_76 = None
        qkv_19 = reshape_38.permute(2, 0, 3, 1, 4)
        reshape_38 = None
        unbind_19 = qkv_19.unbind(0)
        qkv_19 = None
        q_19 = unbind_19[0]
        k_19 = unbind_19[1]
        v_19 = unbind_19[2]
        unbind_19 = None
        x_254 = torch._C._nn.scaled_dot_product_attention(
            q_19, k_19, v_19, attn_mask=None, dropout_p=0.0
        )
        q_19 = k_19 = v_19 = None
        transpose_20 = x_254.transpose(1, 2)
        x_254 = None
        x_255 = transpose_20.reshape(1, 257, 1664)
        transpose_20 = None
        x_256 = torch._C._nn.linear(
            x_255,
            l_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_bias_,
        )
        x_255 = l_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_257 = torch.nn.functional.dropout(x_256, 0.0, False, False)
        x_256 = None
        x_258 = x_252 + x_257
        x_252 = x_257 = None
        x_259 = torch.nn.functional.layer_norm(
            x_258,
            (1664,),
            l_self_modules_blocks_modules_19_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_19_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_19_modules_norm2_parameters_bias_
        ) = None
        x_260 = torch._C._nn.linear(
            x_259,
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_259 = (
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_38 = 1.702 * x_260
        sigmoid_19 = torch.sigmoid(mul_38)
        mul_38 = None
        x_261 = x_260 * sigmoid_19
        x_260 = sigmoid_19 = None
        x_262 = torch.nn.functional.dropout(x_261, 0.0, False, False)
        x_261 = None
        x_263 = torch._C._nn.linear(
            x_262,
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_262 = (
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_264 = torch.nn.functional.dropout(x_263, 0.0, False, False)
        x_263 = None
        x_265 = x_258 + x_264
        x_258 = x_264 = None
        x_266 = torch.nn.functional.layer_norm(
            x_265,
            (1664,),
            l_self_modules_blocks_modules_20_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_20_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_20_modules_norm1_parameters_bias_
        ) = None
        linear_80 = torch._C._nn.linear(
            x_266,
            l_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_bias_,
        )
        x_266 = (
            l_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_40 = linear_80.reshape(1, 257, 3, 16, 104)
        linear_80 = None
        qkv_20 = reshape_40.permute(2, 0, 3, 1, 4)
        reshape_40 = None
        unbind_20 = qkv_20.unbind(0)
        qkv_20 = None
        q_20 = unbind_20[0]
        k_20 = unbind_20[1]
        v_20 = unbind_20[2]
        unbind_20 = None
        x_267 = torch._C._nn.scaled_dot_product_attention(
            q_20, k_20, v_20, attn_mask=None, dropout_p=0.0
        )
        q_20 = k_20 = v_20 = None
        transpose_21 = x_267.transpose(1, 2)
        x_267 = None
        x_268 = transpose_21.reshape(1, 257, 1664)
        transpose_21 = None
        x_269 = torch._C._nn.linear(
            x_268,
            l_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_bias_,
        )
        x_268 = l_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_270 = torch.nn.functional.dropout(x_269, 0.0, False, False)
        x_269 = None
        x_271 = x_265 + x_270
        x_265 = x_270 = None
        x_272 = torch.nn.functional.layer_norm(
            x_271,
            (1664,),
            l_self_modules_blocks_modules_20_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_20_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_20_modules_norm2_parameters_bias_
        ) = None
        x_273 = torch._C._nn.linear(
            x_272,
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_272 = (
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_40 = 1.702 * x_273
        sigmoid_20 = torch.sigmoid(mul_40)
        mul_40 = None
        x_274 = x_273 * sigmoid_20
        x_273 = sigmoid_20 = None
        x_275 = torch.nn.functional.dropout(x_274, 0.0, False, False)
        x_274 = None
        x_276 = torch._C._nn.linear(
            x_275,
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_275 = (
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_277 = torch.nn.functional.dropout(x_276, 0.0, False, False)
        x_276 = None
        x_278 = x_271 + x_277
        x_271 = x_277 = None
        x_279 = torch.nn.functional.layer_norm(
            x_278,
            (1664,),
            l_self_modules_blocks_modules_21_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_21_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_21_modules_norm1_parameters_bias_
        ) = None
        linear_84 = torch._C._nn.linear(
            x_279,
            l_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_bias_,
        )
        x_279 = (
            l_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_42 = linear_84.reshape(1, 257, 3, 16, 104)
        linear_84 = None
        qkv_21 = reshape_42.permute(2, 0, 3, 1, 4)
        reshape_42 = None
        unbind_21 = qkv_21.unbind(0)
        qkv_21 = None
        q_21 = unbind_21[0]
        k_21 = unbind_21[1]
        v_21 = unbind_21[2]
        unbind_21 = None
        x_280 = torch._C._nn.scaled_dot_product_attention(
            q_21, k_21, v_21, attn_mask=None, dropout_p=0.0
        )
        q_21 = k_21 = v_21 = None
        transpose_22 = x_280.transpose(1, 2)
        x_280 = None
        x_281 = transpose_22.reshape(1, 257, 1664)
        transpose_22 = None
        x_282 = torch._C._nn.linear(
            x_281,
            l_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_bias_,
        )
        x_281 = l_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_283 = torch.nn.functional.dropout(x_282, 0.0, False, False)
        x_282 = None
        x_284 = x_278 + x_283
        x_278 = x_283 = None
        x_285 = torch.nn.functional.layer_norm(
            x_284,
            (1664,),
            l_self_modules_blocks_modules_21_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_21_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_21_modules_norm2_parameters_bias_
        ) = None
        x_286 = torch._C._nn.linear(
            x_285,
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_285 = (
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_42 = 1.702 * x_286
        sigmoid_21 = torch.sigmoid(mul_42)
        mul_42 = None
        x_287 = x_286 * sigmoid_21
        x_286 = sigmoid_21 = None
        x_288 = torch.nn.functional.dropout(x_287, 0.0, False, False)
        x_287 = None
        x_289 = torch._C._nn.linear(
            x_288,
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_288 = (
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_290 = torch.nn.functional.dropout(x_289, 0.0, False, False)
        x_289 = None
        x_291 = x_284 + x_290
        x_284 = x_290 = None
        x_292 = torch.nn.functional.layer_norm(
            x_291,
            (1664,),
            l_self_modules_blocks_modules_22_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_22_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_22_modules_norm1_parameters_bias_
        ) = None
        linear_88 = torch._C._nn.linear(
            x_292,
            l_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_bias_,
        )
        x_292 = (
            l_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_44 = linear_88.reshape(1, 257, 3, 16, 104)
        linear_88 = None
        qkv_22 = reshape_44.permute(2, 0, 3, 1, 4)
        reshape_44 = None
        unbind_22 = qkv_22.unbind(0)
        qkv_22 = None
        q_22 = unbind_22[0]
        k_22 = unbind_22[1]
        v_22 = unbind_22[2]
        unbind_22 = None
        x_293 = torch._C._nn.scaled_dot_product_attention(
            q_22, k_22, v_22, attn_mask=None, dropout_p=0.0
        )
        q_22 = k_22 = v_22 = None
        transpose_23 = x_293.transpose(1, 2)
        x_293 = None
        x_294 = transpose_23.reshape(1, 257, 1664)
        transpose_23 = None
        x_295 = torch._C._nn.linear(
            x_294,
            l_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_bias_,
        )
        x_294 = l_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_296 = torch.nn.functional.dropout(x_295, 0.0, False, False)
        x_295 = None
        x_297 = x_291 + x_296
        x_291 = x_296 = None
        x_298 = torch.nn.functional.layer_norm(
            x_297,
            (1664,),
            l_self_modules_blocks_modules_22_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_22_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_22_modules_norm2_parameters_bias_
        ) = None
        x_299 = torch._C._nn.linear(
            x_298,
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_298 = (
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_44 = 1.702 * x_299
        sigmoid_22 = torch.sigmoid(mul_44)
        mul_44 = None
        x_300 = x_299 * sigmoid_22
        x_299 = sigmoid_22 = None
        x_301 = torch.nn.functional.dropout(x_300, 0.0, False, False)
        x_300 = None
        x_302 = torch._C._nn.linear(
            x_301,
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_301 = (
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_303 = torch.nn.functional.dropout(x_302, 0.0, False, False)
        x_302 = None
        x_304 = x_297 + x_303
        x_297 = x_303 = None
        x_305 = torch.nn.functional.layer_norm(
            x_304,
            (1664,),
            l_self_modules_blocks_modules_23_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_23_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_23_modules_norm1_parameters_bias_
        ) = None
        linear_92 = torch._C._nn.linear(
            x_305,
            l_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_bias_,
        )
        x_305 = (
            l_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_46 = linear_92.reshape(1, 257, 3, 16, 104)
        linear_92 = None
        qkv_23 = reshape_46.permute(2, 0, 3, 1, 4)
        reshape_46 = None
        unbind_23 = qkv_23.unbind(0)
        qkv_23 = None
        q_23 = unbind_23[0]
        k_23 = unbind_23[1]
        v_23 = unbind_23[2]
        unbind_23 = None
        x_306 = torch._C._nn.scaled_dot_product_attention(
            q_23, k_23, v_23, attn_mask=None, dropout_p=0.0
        )
        q_23 = k_23 = v_23 = None
        transpose_24 = x_306.transpose(1, 2)
        x_306 = None
        x_307 = transpose_24.reshape(1, 257, 1664)
        transpose_24 = None
        x_308 = torch._C._nn.linear(
            x_307,
            l_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_bias_,
        )
        x_307 = l_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_309 = torch.nn.functional.dropout(x_308, 0.0, False, False)
        x_308 = None
        x_310 = x_304 + x_309
        x_304 = x_309 = None
        x_311 = torch.nn.functional.layer_norm(
            x_310,
            (1664,),
            l_self_modules_blocks_modules_23_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_23_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_23_modules_norm2_parameters_bias_
        ) = None
        x_312 = torch._C._nn.linear(
            x_311,
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_311 = (
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_46 = 1.702 * x_312
        sigmoid_23 = torch.sigmoid(mul_46)
        mul_46 = None
        x_313 = x_312 * sigmoid_23
        x_312 = sigmoid_23 = None
        x_314 = torch.nn.functional.dropout(x_313, 0.0, False, False)
        x_313 = None
        x_315 = torch._C._nn.linear(
            x_314,
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_314 = (
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_316 = torch.nn.functional.dropout(x_315, 0.0, False, False)
        x_315 = None
        x_317 = x_310 + x_316
        x_310 = x_316 = None
        x_318 = torch.nn.functional.layer_norm(
            x_317,
            (1664,),
            l_self_modules_blocks_modules_24_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_24_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_24_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_24_modules_norm1_parameters_bias_
        ) = None
        linear_96 = torch._C._nn.linear(
            x_318,
            l_self_modules_blocks_modules_24_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_24_modules_attn_modules_qkv_parameters_bias_,
        )
        x_318 = (
            l_self_modules_blocks_modules_24_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_24_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_48 = linear_96.reshape(1, 257, 3, 16, 104)
        linear_96 = None
        qkv_24 = reshape_48.permute(2, 0, 3, 1, 4)
        reshape_48 = None
        unbind_24 = qkv_24.unbind(0)
        qkv_24 = None
        q_24 = unbind_24[0]
        k_24 = unbind_24[1]
        v_24 = unbind_24[2]
        unbind_24 = None
        x_319 = torch._C._nn.scaled_dot_product_attention(
            q_24, k_24, v_24, attn_mask=None, dropout_p=0.0
        )
        q_24 = k_24 = v_24 = None
        transpose_25 = x_319.transpose(1, 2)
        x_319 = None
        x_320 = transpose_25.reshape(1, 257, 1664)
        transpose_25 = None
        x_321 = torch._C._nn.linear(
            x_320,
            l_self_modules_blocks_modules_24_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_24_modules_attn_modules_proj_parameters_bias_,
        )
        x_320 = l_self_modules_blocks_modules_24_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_24_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_322 = torch.nn.functional.dropout(x_321, 0.0, False, False)
        x_321 = None
        x_323 = x_317 + x_322
        x_317 = x_322 = None
        x_324 = torch.nn.functional.layer_norm(
            x_323,
            (1664,),
            l_self_modules_blocks_modules_24_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_24_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_24_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_24_modules_norm2_parameters_bias_
        ) = None
        x_325 = torch._C._nn.linear(
            x_324,
            l_self_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_324 = (
            l_self_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_48 = 1.702 * x_325
        sigmoid_24 = torch.sigmoid(mul_48)
        mul_48 = None
        x_326 = x_325 * sigmoid_24
        x_325 = sigmoid_24 = None
        x_327 = torch.nn.functional.dropout(x_326, 0.0, False, False)
        x_326 = None
        x_328 = torch._C._nn.linear(
            x_327,
            l_self_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_327 = (
            l_self_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_329 = torch.nn.functional.dropout(x_328, 0.0, False, False)
        x_328 = None
        x_330 = x_323 + x_329
        x_323 = x_329 = None
        x_331 = torch.nn.functional.layer_norm(
            x_330,
            (1664,),
            l_self_modules_blocks_modules_25_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_25_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_25_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_25_modules_norm1_parameters_bias_
        ) = None
        linear_100 = torch._C._nn.linear(
            x_331,
            l_self_modules_blocks_modules_25_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_25_modules_attn_modules_qkv_parameters_bias_,
        )
        x_331 = (
            l_self_modules_blocks_modules_25_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_25_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_50 = linear_100.reshape(1, 257, 3, 16, 104)
        linear_100 = None
        qkv_25 = reshape_50.permute(2, 0, 3, 1, 4)
        reshape_50 = None
        unbind_25 = qkv_25.unbind(0)
        qkv_25 = None
        q_25 = unbind_25[0]
        k_25 = unbind_25[1]
        v_25 = unbind_25[2]
        unbind_25 = None
        x_332 = torch._C._nn.scaled_dot_product_attention(
            q_25, k_25, v_25, attn_mask=None, dropout_p=0.0
        )
        q_25 = k_25 = v_25 = None
        transpose_26 = x_332.transpose(1, 2)
        x_332 = None
        x_333 = transpose_26.reshape(1, 257, 1664)
        transpose_26 = None
        x_334 = torch._C._nn.linear(
            x_333,
            l_self_modules_blocks_modules_25_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_25_modules_attn_modules_proj_parameters_bias_,
        )
        x_333 = l_self_modules_blocks_modules_25_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_25_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_335 = torch.nn.functional.dropout(x_334, 0.0, False, False)
        x_334 = None
        x_336 = x_330 + x_335
        x_330 = x_335 = None
        x_337 = torch.nn.functional.layer_norm(
            x_336,
            (1664,),
            l_self_modules_blocks_modules_25_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_25_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_25_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_25_modules_norm2_parameters_bias_
        ) = None
        x_338 = torch._C._nn.linear(
            x_337,
            l_self_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_337 = (
            l_self_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_50 = 1.702 * x_338
        sigmoid_25 = torch.sigmoid(mul_50)
        mul_50 = None
        x_339 = x_338 * sigmoid_25
        x_338 = sigmoid_25 = None
        x_340 = torch.nn.functional.dropout(x_339, 0.0, False, False)
        x_339 = None
        x_341 = torch._C._nn.linear(
            x_340,
            l_self_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_340 = (
            l_self_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_342 = torch.nn.functional.dropout(x_341, 0.0, False, False)
        x_341 = None
        x_343 = x_336 + x_342
        x_336 = x_342 = None
        x_344 = torch.nn.functional.layer_norm(
            x_343,
            (1664,),
            l_self_modules_blocks_modules_26_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_26_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_26_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_26_modules_norm1_parameters_bias_
        ) = None
        linear_104 = torch._C._nn.linear(
            x_344,
            l_self_modules_blocks_modules_26_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_26_modules_attn_modules_qkv_parameters_bias_,
        )
        x_344 = (
            l_self_modules_blocks_modules_26_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_26_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_52 = linear_104.reshape(1, 257, 3, 16, 104)
        linear_104 = None
        qkv_26 = reshape_52.permute(2, 0, 3, 1, 4)
        reshape_52 = None
        unbind_26 = qkv_26.unbind(0)
        qkv_26 = None
        q_26 = unbind_26[0]
        k_26 = unbind_26[1]
        v_26 = unbind_26[2]
        unbind_26 = None
        x_345 = torch._C._nn.scaled_dot_product_attention(
            q_26, k_26, v_26, attn_mask=None, dropout_p=0.0
        )
        q_26 = k_26 = v_26 = None
        transpose_27 = x_345.transpose(1, 2)
        x_345 = None
        x_346 = transpose_27.reshape(1, 257, 1664)
        transpose_27 = None
        x_347 = torch._C._nn.linear(
            x_346,
            l_self_modules_blocks_modules_26_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_26_modules_attn_modules_proj_parameters_bias_,
        )
        x_346 = l_self_modules_blocks_modules_26_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_26_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_348 = torch.nn.functional.dropout(x_347, 0.0, False, False)
        x_347 = None
        x_349 = x_343 + x_348
        x_343 = x_348 = None
        x_350 = torch.nn.functional.layer_norm(
            x_349,
            (1664,),
            l_self_modules_blocks_modules_26_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_26_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_26_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_26_modules_norm2_parameters_bias_
        ) = None
        x_351 = torch._C._nn.linear(
            x_350,
            l_self_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_350 = (
            l_self_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_52 = 1.702 * x_351
        sigmoid_26 = torch.sigmoid(mul_52)
        mul_52 = None
        x_352 = x_351 * sigmoid_26
        x_351 = sigmoid_26 = None
        x_353 = torch.nn.functional.dropout(x_352, 0.0, False, False)
        x_352 = None
        x_354 = torch._C._nn.linear(
            x_353,
            l_self_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_353 = (
            l_self_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_355 = torch.nn.functional.dropout(x_354, 0.0, False, False)
        x_354 = None
        x_356 = x_349 + x_355
        x_349 = x_355 = None
        x_357 = torch.nn.functional.layer_norm(
            x_356,
            (1664,),
            l_self_modules_blocks_modules_27_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_27_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_27_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_27_modules_norm1_parameters_bias_
        ) = None
        linear_108 = torch._C._nn.linear(
            x_357,
            l_self_modules_blocks_modules_27_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_27_modules_attn_modules_qkv_parameters_bias_,
        )
        x_357 = (
            l_self_modules_blocks_modules_27_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_27_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_54 = linear_108.reshape(1, 257, 3, 16, 104)
        linear_108 = None
        qkv_27 = reshape_54.permute(2, 0, 3, 1, 4)
        reshape_54 = None
        unbind_27 = qkv_27.unbind(0)
        qkv_27 = None
        q_27 = unbind_27[0]
        k_27 = unbind_27[1]
        v_27 = unbind_27[2]
        unbind_27 = None
        x_358 = torch._C._nn.scaled_dot_product_attention(
            q_27, k_27, v_27, attn_mask=None, dropout_p=0.0
        )
        q_27 = k_27 = v_27 = None
        transpose_28 = x_358.transpose(1, 2)
        x_358 = None
        x_359 = transpose_28.reshape(1, 257, 1664)
        transpose_28 = None
        x_360 = torch._C._nn.linear(
            x_359,
            l_self_modules_blocks_modules_27_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_27_modules_attn_modules_proj_parameters_bias_,
        )
        x_359 = l_self_modules_blocks_modules_27_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_27_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_361 = torch.nn.functional.dropout(x_360, 0.0, False, False)
        x_360 = None
        x_362 = x_356 + x_361
        x_356 = x_361 = None
        x_363 = torch.nn.functional.layer_norm(
            x_362,
            (1664,),
            l_self_modules_blocks_modules_27_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_27_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_27_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_27_modules_norm2_parameters_bias_
        ) = None
        x_364 = torch._C._nn.linear(
            x_363,
            l_self_modules_blocks_modules_27_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_27_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_363 = (
            l_self_modules_blocks_modules_27_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_27_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_54 = 1.702 * x_364
        sigmoid_27 = torch.sigmoid(mul_54)
        mul_54 = None
        x_365 = x_364 * sigmoid_27
        x_364 = sigmoid_27 = None
        x_366 = torch.nn.functional.dropout(x_365, 0.0, False, False)
        x_365 = None
        x_367 = torch._C._nn.linear(
            x_366,
            l_self_modules_blocks_modules_27_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_27_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_366 = (
            l_self_modules_blocks_modules_27_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_27_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_368 = torch.nn.functional.dropout(x_367, 0.0, False, False)
        x_367 = None
        x_369 = x_362 + x_368
        x_362 = x_368 = None
        x_370 = torch.nn.functional.layer_norm(
            x_369,
            (1664,),
            l_self_modules_blocks_modules_28_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_28_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_28_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_28_modules_norm1_parameters_bias_
        ) = None
        linear_112 = torch._C._nn.linear(
            x_370,
            l_self_modules_blocks_modules_28_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_28_modules_attn_modules_qkv_parameters_bias_,
        )
        x_370 = (
            l_self_modules_blocks_modules_28_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_28_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_56 = linear_112.reshape(1, 257, 3, 16, 104)
        linear_112 = None
        qkv_28 = reshape_56.permute(2, 0, 3, 1, 4)
        reshape_56 = None
        unbind_28 = qkv_28.unbind(0)
        qkv_28 = None
        q_28 = unbind_28[0]
        k_28 = unbind_28[1]
        v_28 = unbind_28[2]
        unbind_28 = None
        x_371 = torch._C._nn.scaled_dot_product_attention(
            q_28, k_28, v_28, attn_mask=None, dropout_p=0.0
        )
        q_28 = k_28 = v_28 = None
        transpose_29 = x_371.transpose(1, 2)
        x_371 = None
        x_372 = transpose_29.reshape(1, 257, 1664)
        transpose_29 = None
        x_373 = torch._C._nn.linear(
            x_372,
            l_self_modules_blocks_modules_28_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_28_modules_attn_modules_proj_parameters_bias_,
        )
        x_372 = l_self_modules_blocks_modules_28_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_28_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_374 = torch.nn.functional.dropout(x_373, 0.0, False, False)
        x_373 = None
        x_375 = x_369 + x_374
        x_369 = x_374 = None
        x_376 = torch.nn.functional.layer_norm(
            x_375,
            (1664,),
            l_self_modules_blocks_modules_28_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_28_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_28_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_28_modules_norm2_parameters_bias_
        ) = None
        x_377 = torch._C._nn.linear(
            x_376,
            l_self_modules_blocks_modules_28_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_28_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_376 = (
            l_self_modules_blocks_modules_28_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_28_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_56 = 1.702 * x_377
        sigmoid_28 = torch.sigmoid(mul_56)
        mul_56 = None
        x_378 = x_377 * sigmoid_28
        x_377 = sigmoid_28 = None
        x_379 = torch.nn.functional.dropout(x_378, 0.0, False, False)
        x_378 = None
        x_380 = torch._C._nn.linear(
            x_379,
            l_self_modules_blocks_modules_28_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_28_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_379 = (
            l_self_modules_blocks_modules_28_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_28_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_381 = torch.nn.functional.dropout(x_380, 0.0, False, False)
        x_380 = None
        x_382 = x_375 + x_381
        x_375 = x_381 = None
        x_383 = torch.nn.functional.layer_norm(
            x_382,
            (1664,),
            l_self_modules_blocks_modules_29_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_29_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_29_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_29_modules_norm1_parameters_bias_
        ) = None
        linear_116 = torch._C._nn.linear(
            x_383,
            l_self_modules_blocks_modules_29_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_29_modules_attn_modules_qkv_parameters_bias_,
        )
        x_383 = (
            l_self_modules_blocks_modules_29_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_29_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_58 = linear_116.reshape(1, 257, 3, 16, 104)
        linear_116 = None
        qkv_29 = reshape_58.permute(2, 0, 3, 1, 4)
        reshape_58 = None
        unbind_29 = qkv_29.unbind(0)
        qkv_29 = None
        q_29 = unbind_29[0]
        k_29 = unbind_29[1]
        v_29 = unbind_29[2]
        unbind_29 = None
        x_384 = torch._C._nn.scaled_dot_product_attention(
            q_29, k_29, v_29, attn_mask=None, dropout_p=0.0
        )
        q_29 = k_29 = v_29 = None
        transpose_30 = x_384.transpose(1, 2)
        x_384 = None
        x_385 = transpose_30.reshape(1, 257, 1664)
        transpose_30 = None
        x_386 = torch._C._nn.linear(
            x_385,
            l_self_modules_blocks_modules_29_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_29_modules_attn_modules_proj_parameters_bias_,
        )
        x_385 = l_self_modules_blocks_modules_29_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_29_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_387 = torch.nn.functional.dropout(x_386, 0.0, False, False)
        x_386 = None
        x_388 = x_382 + x_387
        x_382 = x_387 = None
        x_389 = torch.nn.functional.layer_norm(
            x_388,
            (1664,),
            l_self_modules_blocks_modules_29_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_29_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_29_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_29_modules_norm2_parameters_bias_
        ) = None
        x_390 = torch._C._nn.linear(
            x_389,
            l_self_modules_blocks_modules_29_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_29_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_389 = (
            l_self_modules_blocks_modules_29_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_29_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_58 = 1.702 * x_390
        sigmoid_29 = torch.sigmoid(mul_58)
        mul_58 = None
        x_391 = x_390 * sigmoid_29
        x_390 = sigmoid_29 = None
        x_392 = torch.nn.functional.dropout(x_391, 0.0, False, False)
        x_391 = None
        x_393 = torch._C._nn.linear(
            x_392,
            l_self_modules_blocks_modules_29_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_29_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_392 = (
            l_self_modules_blocks_modules_29_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_29_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_394 = torch.nn.functional.dropout(x_393, 0.0, False, False)
        x_393 = None
        x_395 = x_388 + x_394
        x_388 = x_394 = None
        x_396 = torch.nn.functional.layer_norm(
            x_395,
            (1664,),
            l_self_modules_blocks_modules_30_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_30_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_30_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_30_modules_norm1_parameters_bias_
        ) = None
        linear_120 = torch._C._nn.linear(
            x_396,
            l_self_modules_blocks_modules_30_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_30_modules_attn_modules_qkv_parameters_bias_,
        )
        x_396 = (
            l_self_modules_blocks_modules_30_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_30_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_60 = linear_120.reshape(1, 257, 3, 16, 104)
        linear_120 = None
        qkv_30 = reshape_60.permute(2, 0, 3, 1, 4)
        reshape_60 = None
        unbind_30 = qkv_30.unbind(0)
        qkv_30 = None
        q_30 = unbind_30[0]
        k_30 = unbind_30[1]
        v_30 = unbind_30[2]
        unbind_30 = None
        x_397 = torch._C._nn.scaled_dot_product_attention(
            q_30, k_30, v_30, attn_mask=None, dropout_p=0.0
        )
        q_30 = k_30 = v_30 = None
        transpose_31 = x_397.transpose(1, 2)
        x_397 = None
        x_398 = transpose_31.reshape(1, 257, 1664)
        transpose_31 = None
        x_399 = torch._C._nn.linear(
            x_398,
            l_self_modules_blocks_modules_30_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_30_modules_attn_modules_proj_parameters_bias_,
        )
        x_398 = l_self_modules_blocks_modules_30_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_30_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_400 = torch.nn.functional.dropout(x_399, 0.0, False, False)
        x_399 = None
        x_401 = x_395 + x_400
        x_395 = x_400 = None
        x_402 = torch.nn.functional.layer_norm(
            x_401,
            (1664,),
            l_self_modules_blocks_modules_30_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_30_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_30_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_30_modules_norm2_parameters_bias_
        ) = None
        x_403 = torch._C._nn.linear(
            x_402,
            l_self_modules_blocks_modules_30_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_30_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_402 = (
            l_self_modules_blocks_modules_30_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_30_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_60 = 1.702 * x_403
        sigmoid_30 = torch.sigmoid(mul_60)
        mul_60 = None
        x_404 = x_403 * sigmoid_30
        x_403 = sigmoid_30 = None
        x_405 = torch.nn.functional.dropout(x_404, 0.0, False, False)
        x_404 = None
        x_406 = torch._C._nn.linear(
            x_405,
            l_self_modules_blocks_modules_30_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_30_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_405 = (
            l_self_modules_blocks_modules_30_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_30_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_407 = torch.nn.functional.dropout(x_406, 0.0, False, False)
        x_406 = None
        x_408 = x_401 + x_407
        x_401 = x_407 = None
        x_409 = torch.nn.functional.layer_norm(
            x_408,
            (1664,),
            l_self_modules_blocks_modules_31_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_31_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_31_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_31_modules_norm1_parameters_bias_
        ) = None
        linear_124 = torch._C._nn.linear(
            x_409,
            l_self_modules_blocks_modules_31_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_31_modules_attn_modules_qkv_parameters_bias_,
        )
        x_409 = (
            l_self_modules_blocks_modules_31_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_31_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_62 = linear_124.reshape(1, 257, 3, 16, 104)
        linear_124 = None
        qkv_31 = reshape_62.permute(2, 0, 3, 1, 4)
        reshape_62 = None
        unbind_31 = qkv_31.unbind(0)
        qkv_31 = None
        q_31 = unbind_31[0]
        k_31 = unbind_31[1]
        v_31 = unbind_31[2]
        unbind_31 = None
        x_410 = torch._C._nn.scaled_dot_product_attention(
            q_31, k_31, v_31, attn_mask=None, dropout_p=0.0
        )
        q_31 = k_31 = v_31 = None
        transpose_32 = x_410.transpose(1, 2)
        x_410 = None
        x_411 = transpose_32.reshape(1, 257, 1664)
        transpose_32 = None
        x_412 = torch._C._nn.linear(
            x_411,
            l_self_modules_blocks_modules_31_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_31_modules_attn_modules_proj_parameters_bias_,
        )
        x_411 = l_self_modules_blocks_modules_31_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_31_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_413 = torch.nn.functional.dropout(x_412, 0.0, False, False)
        x_412 = None
        x_414 = x_408 + x_413
        x_408 = x_413 = None
        x_415 = torch.nn.functional.layer_norm(
            x_414,
            (1664,),
            l_self_modules_blocks_modules_31_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_31_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_31_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_31_modules_norm2_parameters_bias_
        ) = None
        x_416 = torch._C._nn.linear(
            x_415,
            l_self_modules_blocks_modules_31_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_31_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_415 = (
            l_self_modules_blocks_modules_31_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_31_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_62 = 1.702 * x_416
        sigmoid_31 = torch.sigmoid(mul_62)
        mul_62 = None
        x_417 = x_416 * sigmoid_31
        x_416 = sigmoid_31 = None
        x_418 = torch.nn.functional.dropout(x_417, 0.0, False, False)
        x_417 = None
        x_419 = torch._C._nn.linear(
            x_418,
            l_self_modules_blocks_modules_31_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_31_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_418 = (
            l_self_modules_blocks_modules_31_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_31_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_420 = torch.nn.functional.dropout(x_419, 0.0, False, False)
        x_419 = None
        x_421 = x_414 + x_420
        x_414 = x_420 = None
        x_422 = torch.nn.functional.layer_norm(
            x_421,
            (1664,),
            l_self_modules_blocks_modules_32_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_32_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_32_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_32_modules_norm1_parameters_bias_
        ) = None
        linear_128 = torch._C._nn.linear(
            x_422,
            l_self_modules_blocks_modules_32_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_32_modules_attn_modules_qkv_parameters_bias_,
        )
        x_422 = (
            l_self_modules_blocks_modules_32_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_32_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_64 = linear_128.reshape(1, 257, 3, 16, 104)
        linear_128 = None
        qkv_32 = reshape_64.permute(2, 0, 3, 1, 4)
        reshape_64 = None
        unbind_32 = qkv_32.unbind(0)
        qkv_32 = None
        q_32 = unbind_32[0]
        k_32 = unbind_32[1]
        v_32 = unbind_32[2]
        unbind_32 = None
        x_423 = torch._C._nn.scaled_dot_product_attention(
            q_32, k_32, v_32, attn_mask=None, dropout_p=0.0
        )
        q_32 = k_32 = v_32 = None
        transpose_33 = x_423.transpose(1, 2)
        x_423 = None
        x_424 = transpose_33.reshape(1, 257, 1664)
        transpose_33 = None
        x_425 = torch._C._nn.linear(
            x_424,
            l_self_modules_blocks_modules_32_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_32_modules_attn_modules_proj_parameters_bias_,
        )
        x_424 = l_self_modules_blocks_modules_32_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_32_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_426 = torch.nn.functional.dropout(x_425, 0.0, False, False)
        x_425 = None
        x_427 = x_421 + x_426
        x_421 = x_426 = None
        x_428 = torch.nn.functional.layer_norm(
            x_427,
            (1664,),
            l_self_modules_blocks_modules_32_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_32_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_32_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_32_modules_norm2_parameters_bias_
        ) = None
        x_429 = torch._C._nn.linear(
            x_428,
            l_self_modules_blocks_modules_32_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_32_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_428 = (
            l_self_modules_blocks_modules_32_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_32_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_64 = 1.702 * x_429
        sigmoid_32 = torch.sigmoid(mul_64)
        mul_64 = None
        x_430 = x_429 * sigmoid_32
        x_429 = sigmoid_32 = None
        x_431 = torch.nn.functional.dropout(x_430, 0.0, False, False)
        x_430 = None
        x_432 = torch._C._nn.linear(
            x_431,
            l_self_modules_blocks_modules_32_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_32_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_431 = (
            l_self_modules_blocks_modules_32_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_32_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_433 = torch.nn.functional.dropout(x_432, 0.0, False, False)
        x_432 = None
        x_434 = x_427 + x_433
        x_427 = x_433 = None
        x_435 = torch.nn.functional.layer_norm(
            x_434,
            (1664,),
            l_self_modules_blocks_modules_33_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_33_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_33_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_33_modules_norm1_parameters_bias_
        ) = None
        linear_132 = torch._C._nn.linear(
            x_435,
            l_self_modules_blocks_modules_33_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_33_modules_attn_modules_qkv_parameters_bias_,
        )
        x_435 = (
            l_self_modules_blocks_modules_33_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_33_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_66 = linear_132.reshape(1, 257, 3, 16, 104)
        linear_132 = None
        qkv_33 = reshape_66.permute(2, 0, 3, 1, 4)
        reshape_66 = None
        unbind_33 = qkv_33.unbind(0)
        qkv_33 = None
        q_33 = unbind_33[0]
        k_33 = unbind_33[1]
        v_33 = unbind_33[2]
        unbind_33 = None
        x_436 = torch._C._nn.scaled_dot_product_attention(
            q_33, k_33, v_33, attn_mask=None, dropout_p=0.0
        )
        q_33 = k_33 = v_33 = None
        transpose_34 = x_436.transpose(1, 2)
        x_436 = None
        x_437 = transpose_34.reshape(1, 257, 1664)
        transpose_34 = None
        x_438 = torch._C._nn.linear(
            x_437,
            l_self_modules_blocks_modules_33_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_33_modules_attn_modules_proj_parameters_bias_,
        )
        x_437 = l_self_modules_blocks_modules_33_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_33_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_439 = torch.nn.functional.dropout(x_438, 0.0, False, False)
        x_438 = None
        x_440 = x_434 + x_439
        x_434 = x_439 = None
        x_441 = torch.nn.functional.layer_norm(
            x_440,
            (1664,),
            l_self_modules_blocks_modules_33_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_33_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_33_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_33_modules_norm2_parameters_bias_
        ) = None
        x_442 = torch._C._nn.linear(
            x_441,
            l_self_modules_blocks_modules_33_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_33_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_441 = (
            l_self_modules_blocks_modules_33_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_33_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_66 = 1.702 * x_442
        sigmoid_33 = torch.sigmoid(mul_66)
        mul_66 = None
        x_443 = x_442 * sigmoid_33
        x_442 = sigmoid_33 = None
        x_444 = torch.nn.functional.dropout(x_443, 0.0, False, False)
        x_443 = None
        x_445 = torch._C._nn.linear(
            x_444,
            l_self_modules_blocks_modules_33_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_33_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_444 = (
            l_self_modules_blocks_modules_33_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_33_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_446 = torch.nn.functional.dropout(x_445, 0.0, False, False)
        x_445 = None
        x_447 = x_440 + x_446
        x_440 = x_446 = None
        x_448 = torch.nn.functional.layer_norm(
            x_447,
            (1664,),
            l_self_modules_blocks_modules_34_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_34_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_34_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_34_modules_norm1_parameters_bias_
        ) = None
        linear_136 = torch._C._nn.linear(
            x_448,
            l_self_modules_blocks_modules_34_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_34_modules_attn_modules_qkv_parameters_bias_,
        )
        x_448 = (
            l_self_modules_blocks_modules_34_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_34_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_68 = linear_136.reshape(1, 257, 3, 16, 104)
        linear_136 = None
        qkv_34 = reshape_68.permute(2, 0, 3, 1, 4)
        reshape_68 = None
        unbind_34 = qkv_34.unbind(0)
        qkv_34 = None
        q_34 = unbind_34[0]
        k_34 = unbind_34[1]
        v_34 = unbind_34[2]
        unbind_34 = None
        x_449 = torch._C._nn.scaled_dot_product_attention(
            q_34, k_34, v_34, attn_mask=None, dropout_p=0.0
        )
        q_34 = k_34 = v_34 = None
        transpose_35 = x_449.transpose(1, 2)
        x_449 = None
        x_450 = transpose_35.reshape(1, 257, 1664)
        transpose_35 = None
        x_451 = torch._C._nn.linear(
            x_450,
            l_self_modules_blocks_modules_34_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_34_modules_attn_modules_proj_parameters_bias_,
        )
        x_450 = l_self_modules_blocks_modules_34_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_34_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_452 = torch.nn.functional.dropout(x_451, 0.0, False, False)
        x_451 = None
        x_453 = x_447 + x_452
        x_447 = x_452 = None
        x_454 = torch.nn.functional.layer_norm(
            x_453,
            (1664,),
            l_self_modules_blocks_modules_34_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_34_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_34_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_34_modules_norm2_parameters_bias_
        ) = None
        x_455 = torch._C._nn.linear(
            x_454,
            l_self_modules_blocks_modules_34_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_34_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_454 = (
            l_self_modules_blocks_modules_34_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_34_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_68 = 1.702 * x_455
        sigmoid_34 = torch.sigmoid(mul_68)
        mul_68 = None
        x_456 = x_455 * sigmoid_34
        x_455 = sigmoid_34 = None
        x_457 = torch.nn.functional.dropout(x_456, 0.0, False, False)
        x_456 = None
        x_458 = torch._C._nn.linear(
            x_457,
            l_self_modules_blocks_modules_34_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_34_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_457 = (
            l_self_modules_blocks_modules_34_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_34_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_459 = torch.nn.functional.dropout(x_458, 0.0, False, False)
        x_458 = None
        x_460 = x_453 + x_459
        x_453 = x_459 = None
        x_461 = torch.nn.functional.layer_norm(
            x_460,
            (1664,),
            l_self_modules_blocks_modules_35_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_35_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_35_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_35_modules_norm1_parameters_bias_
        ) = None
        linear_140 = torch._C._nn.linear(
            x_461,
            l_self_modules_blocks_modules_35_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_35_modules_attn_modules_qkv_parameters_bias_,
        )
        x_461 = (
            l_self_modules_blocks_modules_35_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_35_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_70 = linear_140.reshape(1, 257, 3, 16, 104)
        linear_140 = None
        qkv_35 = reshape_70.permute(2, 0, 3, 1, 4)
        reshape_70 = None
        unbind_35 = qkv_35.unbind(0)
        qkv_35 = None
        q_35 = unbind_35[0]
        k_35 = unbind_35[1]
        v_35 = unbind_35[2]
        unbind_35 = None
        x_462 = torch._C._nn.scaled_dot_product_attention(
            q_35, k_35, v_35, attn_mask=None, dropout_p=0.0
        )
        q_35 = k_35 = v_35 = None
        transpose_36 = x_462.transpose(1, 2)
        x_462 = None
        x_463 = transpose_36.reshape(1, 257, 1664)
        transpose_36 = None
        x_464 = torch._C._nn.linear(
            x_463,
            l_self_modules_blocks_modules_35_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_35_modules_attn_modules_proj_parameters_bias_,
        )
        x_463 = l_self_modules_blocks_modules_35_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_35_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_465 = torch.nn.functional.dropout(x_464, 0.0, False, False)
        x_464 = None
        x_466 = x_460 + x_465
        x_460 = x_465 = None
        x_467 = torch.nn.functional.layer_norm(
            x_466,
            (1664,),
            l_self_modules_blocks_modules_35_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_35_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_35_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_35_modules_norm2_parameters_bias_
        ) = None
        x_468 = torch._C._nn.linear(
            x_467,
            l_self_modules_blocks_modules_35_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_35_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_467 = (
            l_self_modules_blocks_modules_35_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_35_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_70 = 1.702 * x_468
        sigmoid_35 = torch.sigmoid(mul_70)
        mul_70 = None
        x_469 = x_468 * sigmoid_35
        x_468 = sigmoid_35 = None
        x_470 = torch.nn.functional.dropout(x_469, 0.0, False, False)
        x_469 = None
        x_471 = torch._C._nn.linear(
            x_470,
            l_self_modules_blocks_modules_35_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_35_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_470 = (
            l_self_modules_blocks_modules_35_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_35_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_472 = torch.nn.functional.dropout(x_471, 0.0, False, False)
        x_471 = None
        x_473 = x_466 + x_472
        x_466 = x_472 = None
        x_474 = torch.nn.functional.layer_norm(
            x_473,
            (1664,),
            l_self_modules_blocks_modules_36_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_36_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_36_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_36_modules_norm1_parameters_bias_
        ) = None
        linear_144 = torch._C._nn.linear(
            x_474,
            l_self_modules_blocks_modules_36_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_36_modules_attn_modules_qkv_parameters_bias_,
        )
        x_474 = (
            l_self_modules_blocks_modules_36_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_36_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_72 = linear_144.reshape(1, 257, 3, 16, 104)
        linear_144 = None
        qkv_36 = reshape_72.permute(2, 0, 3, 1, 4)
        reshape_72 = None
        unbind_36 = qkv_36.unbind(0)
        qkv_36 = None
        q_36 = unbind_36[0]
        k_36 = unbind_36[1]
        v_36 = unbind_36[2]
        unbind_36 = None
        x_475 = torch._C._nn.scaled_dot_product_attention(
            q_36, k_36, v_36, attn_mask=None, dropout_p=0.0
        )
        q_36 = k_36 = v_36 = None
        transpose_37 = x_475.transpose(1, 2)
        x_475 = None
        x_476 = transpose_37.reshape(1, 257, 1664)
        transpose_37 = None
        x_477 = torch._C._nn.linear(
            x_476,
            l_self_modules_blocks_modules_36_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_36_modules_attn_modules_proj_parameters_bias_,
        )
        x_476 = l_self_modules_blocks_modules_36_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_36_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_478 = torch.nn.functional.dropout(x_477, 0.0, False, False)
        x_477 = None
        x_479 = x_473 + x_478
        x_473 = x_478 = None
        x_480 = torch.nn.functional.layer_norm(
            x_479,
            (1664,),
            l_self_modules_blocks_modules_36_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_36_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_36_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_36_modules_norm2_parameters_bias_
        ) = None
        x_481 = torch._C._nn.linear(
            x_480,
            l_self_modules_blocks_modules_36_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_36_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_480 = (
            l_self_modules_blocks_modules_36_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_36_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_72 = 1.702 * x_481
        sigmoid_36 = torch.sigmoid(mul_72)
        mul_72 = None
        x_482 = x_481 * sigmoid_36
        x_481 = sigmoid_36 = None
        x_483 = torch.nn.functional.dropout(x_482, 0.0, False, False)
        x_482 = None
        x_484 = torch._C._nn.linear(
            x_483,
            l_self_modules_blocks_modules_36_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_36_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_483 = (
            l_self_modules_blocks_modules_36_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_36_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_485 = torch.nn.functional.dropout(x_484, 0.0, False, False)
        x_484 = None
        x_486 = x_479 + x_485
        x_479 = x_485 = None
        x_487 = torch.nn.functional.layer_norm(
            x_486,
            (1664,),
            l_self_modules_blocks_modules_37_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_37_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_37_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_37_modules_norm1_parameters_bias_
        ) = None
        linear_148 = torch._C._nn.linear(
            x_487,
            l_self_modules_blocks_modules_37_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_37_modules_attn_modules_qkv_parameters_bias_,
        )
        x_487 = (
            l_self_modules_blocks_modules_37_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_37_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_74 = linear_148.reshape(1, 257, 3, 16, 104)
        linear_148 = None
        qkv_37 = reshape_74.permute(2, 0, 3, 1, 4)
        reshape_74 = None
        unbind_37 = qkv_37.unbind(0)
        qkv_37 = None
        q_37 = unbind_37[0]
        k_37 = unbind_37[1]
        v_37 = unbind_37[2]
        unbind_37 = None
        x_488 = torch._C._nn.scaled_dot_product_attention(
            q_37, k_37, v_37, attn_mask=None, dropout_p=0.0
        )
        q_37 = k_37 = v_37 = None
        transpose_38 = x_488.transpose(1, 2)
        x_488 = None
        x_489 = transpose_38.reshape(1, 257, 1664)
        transpose_38 = None
        x_490 = torch._C._nn.linear(
            x_489,
            l_self_modules_blocks_modules_37_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_37_modules_attn_modules_proj_parameters_bias_,
        )
        x_489 = l_self_modules_blocks_modules_37_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_37_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_491 = torch.nn.functional.dropout(x_490, 0.0, False, False)
        x_490 = None
        x_492 = x_486 + x_491
        x_486 = x_491 = None
        x_493 = torch.nn.functional.layer_norm(
            x_492,
            (1664,),
            l_self_modules_blocks_modules_37_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_37_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_37_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_37_modules_norm2_parameters_bias_
        ) = None
        x_494 = torch._C._nn.linear(
            x_493,
            l_self_modules_blocks_modules_37_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_37_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_493 = (
            l_self_modules_blocks_modules_37_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_37_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_74 = 1.702 * x_494
        sigmoid_37 = torch.sigmoid(mul_74)
        mul_74 = None
        x_495 = x_494 * sigmoid_37
        x_494 = sigmoid_37 = None
        x_496 = torch.nn.functional.dropout(x_495, 0.0, False, False)
        x_495 = None
        x_497 = torch._C._nn.linear(
            x_496,
            l_self_modules_blocks_modules_37_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_37_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_496 = (
            l_self_modules_blocks_modules_37_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_37_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_498 = torch.nn.functional.dropout(x_497, 0.0, False, False)
        x_497 = None
        x_499 = x_492 + x_498
        x_492 = x_498 = None
        x_500 = torch.nn.functional.layer_norm(
            x_499,
            (1664,),
            l_self_modules_blocks_modules_38_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_38_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_38_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_38_modules_norm1_parameters_bias_
        ) = None
        linear_152 = torch._C._nn.linear(
            x_500,
            l_self_modules_blocks_modules_38_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_38_modules_attn_modules_qkv_parameters_bias_,
        )
        x_500 = (
            l_self_modules_blocks_modules_38_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_38_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_76 = linear_152.reshape(1, 257, 3, 16, 104)
        linear_152 = None
        qkv_38 = reshape_76.permute(2, 0, 3, 1, 4)
        reshape_76 = None
        unbind_38 = qkv_38.unbind(0)
        qkv_38 = None
        q_38 = unbind_38[0]
        k_38 = unbind_38[1]
        v_38 = unbind_38[2]
        unbind_38 = None
        x_501 = torch._C._nn.scaled_dot_product_attention(
            q_38, k_38, v_38, attn_mask=None, dropout_p=0.0
        )
        q_38 = k_38 = v_38 = None
        transpose_39 = x_501.transpose(1, 2)
        x_501 = None
        x_502 = transpose_39.reshape(1, 257, 1664)
        transpose_39 = None
        x_503 = torch._C._nn.linear(
            x_502,
            l_self_modules_blocks_modules_38_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_38_modules_attn_modules_proj_parameters_bias_,
        )
        x_502 = l_self_modules_blocks_modules_38_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_38_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_504 = torch.nn.functional.dropout(x_503, 0.0, False, False)
        x_503 = None
        x_505 = x_499 + x_504
        x_499 = x_504 = None
        x_506 = torch.nn.functional.layer_norm(
            x_505,
            (1664,),
            l_self_modules_blocks_modules_38_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_38_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_38_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_38_modules_norm2_parameters_bias_
        ) = None
        x_507 = torch._C._nn.linear(
            x_506,
            l_self_modules_blocks_modules_38_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_38_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_506 = (
            l_self_modules_blocks_modules_38_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_38_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_76 = 1.702 * x_507
        sigmoid_38 = torch.sigmoid(mul_76)
        mul_76 = None
        x_508 = x_507 * sigmoid_38
        x_507 = sigmoid_38 = None
        x_509 = torch.nn.functional.dropout(x_508, 0.0, False, False)
        x_508 = None
        x_510 = torch._C._nn.linear(
            x_509,
            l_self_modules_blocks_modules_38_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_38_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_509 = (
            l_self_modules_blocks_modules_38_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_38_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_511 = torch.nn.functional.dropout(x_510, 0.0, False, False)
        x_510 = None
        x_512 = x_505 + x_511
        x_505 = x_511 = None
        x_513 = torch.nn.functional.layer_norm(
            x_512,
            (1664,),
            l_self_modules_blocks_modules_39_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_39_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_39_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_39_modules_norm1_parameters_bias_
        ) = None
        linear_156 = torch._C._nn.linear(
            x_513,
            l_self_modules_blocks_modules_39_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_39_modules_attn_modules_qkv_parameters_bias_,
        )
        x_513 = (
            l_self_modules_blocks_modules_39_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_39_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_78 = linear_156.reshape(1, 257, 3, 16, 104)
        linear_156 = None
        qkv_39 = reshape_78.permute(2, 0, 3, 1, 4)
        reshape_78 = None
        unbind_39 = qkv_39.unbind(0)
        qkv_39 = None
        q_39 = unbind_39[0]
        k_39 = unbind_39[1]
        v_39 = unbind_39[2]
        unbind_39 = None
        x_514 = torch._C._nn.scaled_dot_product_attention(
            q_39, k_39, v_39, attn_mask=None, dropout_p=0.0
        )
        q_39 = k_39 = v_39 = None
        transpose_40 = x_514.transpose(1, 2)
        x_514 = None
        x_515 = transpose_40.reshape(1, 257, 1664)
        transpose_40 = None
        x_516 = torch._C._nn.linear(
            x_515,
            l_self_modules_blocks_modules_39_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_39_modules_attn_modules_proj_parameters_bias_,
        )
        x_515 = l_self_modules_blocks_modules_39_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_39_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_517 = torch.nn.functional.dropout(x_516, 0.0, False, False)
        x_516 = None
        x_518 = x_512 + x_517
        x_512 = x_517 = None
        x_519 = torch.nn.functional.layer_norm(
            x_518,
            (1664,),
            l_self_modules_blocks_modules_39_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_39_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_39_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_39_modules_norm2_parameters_bias_
        ) = None
        x_520 = torch._C._nn.linear(
            x_519,
            l_self_modules_blocks_modules_39_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_39_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_519 = (
            l_self_modules_blocks_modules_39_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_39_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_78 = 1.702 * x_520
        sigmoid_39 = torch.sigmoid(mul_78)
        mul_78 = None
        x_521 = x_520 * sigmoid_39
        x_520 = sigmoid_39 = None
        x_522 = torch.nn.functional.dropout(x_521, 0.0, False, False)
        x_521 = None
        x_523 = torch._C._nn.linear(
            x_522,
            l_self_modules_blocks_modules_39_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_39_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_522 = (
            l_self_modules_blocks_modules_39_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_39_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_524 = torch.nn.functional.dropout(x_523, 0.0, False, False)
        x_523 = None
        x_525 = x_518 + x_524
        x_518 = x_524 = None
        x_526 = torch.nn.functional.layer_norm(
            x_525,
            (1664,),
            l_self_modules_blocks_modules_40_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_40_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_40_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_40_modules_norm1_parameters_bias_
        ) = None
        linear_160 = torch._C._nn.linear(
            x_526,
            l_self_modules_blocks_modules_40_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_40_modules_attn_modules_qkv_parameters_bias_,
        )
        x_526 = (
            l_self_modules_blocks_modules_40_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_40_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_80 = linear_160.reshape(1, 257, 3, 16, 104)
        linear_160 = None
        qkv_40 = reshape_80.permute(2, 0, 3, 1, 4)
        reshape_80 = None
        unbind_40 = qkv_40.unbind(0)
        qkv_40 = None
        q_40 = unbind_40[0]
        k_40 = unbind_40[1]
        v_40 = unbind_40[2]
        unbind_40 = None
        x_527 = torch._C._nn.scaled_dot_product_attention(
            q_40, k_40, v_40, attn_mask=None, dropout_p=0.0
        )
        q_40 = k_40 = v_40 = None
        transpose_41 = x_527.transpose(1, 2)
        x_527 = None
        x_528 = transpose_41.reshape(1, 257, 1664)
        transpose_41 = None
        x_529 = torch._C._nn.linear(
            x_528,
            l_self_modules_blocks_modules_40_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_40_modules_attn_modules_proj_parameters_bias_,
        )
        x_528 = l_self_modules_blocks_modules_40_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_40_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_530 = torch.nn.functional.dropout(x_529, 0.0, False, False)
        x_529 = None
        x_531 = x_525 + x_530
        x_525 = x_530 = None
        x_532 = torch.nn.functional.layer_norm(
            x_531,
            (1664,),
            l_self_modules_blocks_modules_40_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_40_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_40_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_40_modules_norm2_parameters_bias_
        ) = None
        x_533 = torch._C._nn.linear(
            x_532,
            l_self_modules_blocks_modules_40_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_40_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_532 = (
            l_self_modules_blocks_modules_40_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_40_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_80 = 1.702 * x_533
        sigmoid_40 = torch.sigmoid(mul_80)
        mul_80 = None
        x_534 = x_533 * sigmoid_40
        x_533 = sigmoid_40 = None
        x_535 = torch.nn.functional.dropout(x_534, 0.0, False, False)
        x_534 = None
        x_536 = torch._C._nn.linear(
            x_535,
            l_self_modules_blocks_modules_40_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_40_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_535 = (
            l_self_modules_blocks_modules_40_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_40_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_537 = torch.nn.functional.dropout(x_536, 0.0, False, False)
        x_536 = None
        x_538 = x_531 + x_537
        x_531 = x_537 = None
        x_539 = torch.nn.functional.layer_norm(
            x_538,
            (1664,),
            l_self_modules_blocks_modules_41_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_41_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_41_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_41_modules_norm1_parameters_bias_
        ) = None
        linear_164 = torch._C._nn.linear(
            x_539,
            l_self_modules_blocks_modules_41_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_41_modules_attn_modules_qkv_parameters_bias_,
        )
        x_539 = (
            l_self_modules_blocks_modules_41_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_41_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_82 = linear_164.reshape(1, 257, 3, 16, 104)
        linear_164 = None
        qkv_41 = reshape_82.permute(2, 0, 3, 1, 4)
        reshape_82 = None
        unbind_41 = qkv_41.unbind(0)
        qkv_41 = None
        q_41 = unbind_41[0]
        k_41 = unbind_41[1]
        v_41 = unbind_41[2]
        unbind_41 = None
        x_540 = torch._C._nn.scaled_dot_product_attention(
            q_41, k_41, v_41, attn_mask=None, dropout_p=0.0
        )
        q_41 = k_41 = v_41 = None
        transpose_42 = x_540.transpose(1, 2)
        x_540 = None
        x_541 = transpose_42.reshape(1, 257, 1664)
        transpose_42 = None
        x_542 = torch._C._nn.linear(
            x_541,
            l_self_modules_blocks_modules_41_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_41_modules_attn_modules_proj_parameters_bias_,
        )
        x_541 = l_self_modules_blocks_modules_41_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_41_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_543 = torch.nn.functional.dropout(x_542, 0.0, False, False)
        x_542 = None
        x_544 = x_538 + x_543
        x_538 = x_543 = None
        x_545 = torch.nn.functional.layer_norm(
            x_544,
            (1664,),
            l_self_modules_blocks_modules_41_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_41_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_41_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_41_modules_norm2_parameters_bias_
        ) = None
        x_546 = torch._C._nn.linear(
            x_545,
            l_self_modules_blocks_modules_41_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_41_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_545 = (
            l_self_modules_blocks_modules_41_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_41_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_82 = 1.702 * x_546
        sigmoid_41 = torch.sigmoid(mul_82)
        mul_82 = None
        x_547 = x_546 * sigmoid_41
        x_546 = sigmoid_41 = None
        x_548 = torch.nn.functional.dropout(x_547, 0.0, False, False)
        x_547 = None
        x_549 = torch._C._nn.linear(
            x_548,
            l_self_modules_blocks_modules_41_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_41_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_548 = (
            l_self_modules_blocks_modules_41_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_41_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_550 = torch.nn.functional.dropout(x_549, 0.0, False, False)
        x_549 = None
        x_551 = x_544 + x_550
        x_544 = x_550 = None
        x_552 = torch.nn.functional.layer_norm(
            x_551,
            (1664,),
            l_self_modules_blocks_modules_42_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_42_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_42_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_42_modules_norm1_parameters_bias_
        ) = None
        linear_168 = torch._C._nn.linear(
            x_552,
            l_self_modules_blocks_modules_42_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_42_modules_attn_modules_qkv_parameters_bias_,
        )
        x_552 = (
            l_self_modules_blocks_modules_42_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_42_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_84 = linear_168.reshape(1, 257, 3, 16, 104)
        linear_168 = None
        qkv_42 = reshape_84.permute(2, 0, 3, 1, 4)
        reshape_84 = None
        unbind_42 = qkv_42.unbind(0)
        qkv_42 = None
        q_42 = unbind_42[0]
        k_42 = unbind_42[1]
        v_42 = unbind_42[2]
        unbind_42 = None
        x_553 = torch._C._nn.scaled_dot_product_attention(
            q_42, k_42, v_42, attn_mask=None, dropout_p=0.0
        )
        q_42 = k_42 = v_42 = None
        transpose_43 = x_553.transpose(1, 2)
        x_553 = None
        x_554 = transpose_43.reshape(1, 257, 1664)
        transpose_43 = None
        x_555 = torch._C._nn.linear(
            x_554,
            l_self_modules_blocks_modules_42_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_42_modules_attn_modules_proj_parameters_bias_,
        )
        x_554 = l_self_modules_blocks_modules_42_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_42_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_556 = torch.nn.functional.dropout(x_555, 0.0, False, False)
        x_555 = None
        x_557 = x_551 + x_556
        x_551 = x_556 = None
        x_558 = torch.nn.functional.layer_norm(
            x_557,
            (1664,),
            l_self_modules_blocks_modules_42_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_42_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_42_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_42_modules_norm2_parameters_bias_
        ) = None
        x_559 = torch._C._nn.linear(
            x_558,
            l_self_modules_blocks_modules_42_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_42_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_558 = (
            l_self_modules_blocks_modules_42_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_42_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_84 = 1.702 * x_559
        sigmoid_42 = torch.sigmoid(mul_84)
        mul_84 = None
        x_560 = x_559 * sigmoid_42
        x_559 = sigmoid_42 = None
        x_561 = torch.nn.functional.dropout(x_560, 0.0, False, False)
        x_560 = None
        x_562 = torch._C._nn.linear(
            x_561,
            l_self_modules_blocks_modules_42_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_42_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_561 = (
            l_self_modules_blocks_modules_42_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_42_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_563 = torch.nn.functional.dropout(x_562, 0.0, False, False)
        x_562 = None
        x_564 = x_557 + x_563
        x_557 = x_563 = None
        x_565 = torch.nn.functional.layer_norm(
            x_564,
            (1664,),
            l_self_modules_blocks_modules_43_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_43_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_43_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_43_modules_norm1_parameters_bias_
        ) = None
        linear_172 = torch._C._nn.linear(
            x_565,
            l_self_modules_blocks_modules_43_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_43_modules_attn_modules_qkv_parameters_bias_,
        )
        x_565 = (
            l_self_modules_blocks_modules_43_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_43_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_86 = linear_172.reshape(1, 257, 3, 16, 104)
        linear_172 = None
        qkv_43 = reshape_86.permute(2, 0, 3, 1, 4)
        reshape_86 = None
        unbind_43 = qkv_43.unbind(0)
        qkv_43 = None
        q_43 = unbind_43[0]
        k_43 = unbind_43[1]
        v_43 = unbind_43[2]
        unbind_43 = None
        x_566 = torch._C._nn.scaled_dot_product_attention(
            q_43, k_43, v_43, attn_mask=None, dropout_p=0.0
        )
        q_43 = k_43 = v_43 = None
        transpose_44 = x_566.transpose(1, 2)
        x_566 = None
        x_567 = transpose_44.reshape(1, 257, 1664)
        transpose_44 = None
        x_568 = torch._C._nn.linear(
            x_567,
            l_self_modules_blocks_modules_43_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_43_modules_attn_modules_proj_parameters_bias_,
        )
        x_567 = l_self_modules_blocks_modules_43_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_43_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_569 = torch.nn.functional.dropout(x_568, 0.0, False, False)
        x_568 = None
        x_570 = x_564 + x_569
        x_564 = x_569 = None
        x_571 = torch.nn.functional.layer_norm(
            x_570,
            (1664,),
            l_self_modules_blocks_modules_43_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_43_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_43_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_43_modules_norm2_parameters_bias_
        ) = None
        x_572 = torch._C._nn.linear(
            x_571,
            l_self_modules_blocks_modules_43_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_43_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_571 = (
            l_self_modules_blocks_modules_43_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_43_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_86 = 1.702 * x_572
        sigmoid_43 = torch.sigmoid(mul_86)
        mul_86 = None
        x_573 = x_572 * sigmoid_43
        x_572 = sigmoid_43 = None
        x_574 = torch.nn.functional.dropout(x_573, 0.0, False, False)
        x_573 = None
        x_575 = torch._C._nn.linear(
            x_574,
            l_self_modules_blocks_modules_43_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_43_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_574 = (
            l_self_modules_blocks_modules_43_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_43_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_576 = torch.nn.functional.dropout(x_575, 0.0, False, False)
        x_575 = None
        x_577 = x_570 + x_576
        x_570 = x_576 = None
        x_578 = torch.nn.functional.layer_norm(
            x_577,
            (1664,),
            l_self_modules_blocks_modules_44_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_44_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_44_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_44_modules_norm1_parameters_bias_
        ) = None
        linear_176 = torch._C._nn.linear(
            x_578,
            l_self_modules_blocks_modules_44_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_44_modules_attn_modules_qkv_parameters_bias_,
        )
        x_578 = (
            l_self_modules_blocks_modules_44_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_44_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_88 = linear_176.reshape(1, 257, 3, 16, 104)
        linear_176 = None
        qkv_44 = reshape_88.permute(2, 0, 3, 1, 4)
        reshape_88 = None
        unbind_44 = qkv_44.unbind(0)
        qkv_44 = None
        q_44 = unbind_44[0]
        k_44 = unbind_44[1]
        v_44 = unbind_44[2]
        unbind_44 = None
        x_579 = torch._C._nn.scaled_dot_product_attention(
            q_44, k_44, v_44, attn_mask=None, dropout_p=0.0
        )
        q_44 = k_44 = v_44 = None
        transpose_45 = x_579.transpose(1, 2)
        x_579 = None
        x_580 = transpose_45.reshape(1, 257, 1664)
        transpose_45 = None
        x_581 = torch._C._nn.linear(
            x_580,
            l_self_modules_blocks_modules_44_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_44_modules_attn_modules_proj_parameters_bias_,
        )
        x_580 = l_self_modules_blocks_modules_44_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_44_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_582 = torch.nn.functional.dropout(x_581, 0.0, False, False)
        x_581 = None
        x_583 = x_577 + x_582
        x_577 = x_582 = None
        x_584 = torch.nn.functional.layer_norm(
            x_583,
            (1664,),
            l_self_modules_blocks_modules_44_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_44_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_44_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_44_modules_norm2_parameters_bias_
        ) = None
        x_585 = torch._C._nn.linear(
            x_584,
            l_self_modules_blocks_modules_44_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_44_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_584 = (
            l_self_modules_blocks_modules_44_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_44_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_88 = 1.702 * x_585
        sigmoid_44 = torch.sigmoid(mul_88)
        mul_88 = None
        x_586 = x_585 * sigmoid_44
        x_585 = sigmoid_44 = None
        x_587 = torch.nn.functional.dropout(x_586, 0.0, False, False)
        x_586 = None
        x_588 = torch._C._nn.linear(
            x_587,
            l_self_modules_blocks_modules_44_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_44_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_587 = (
            l_self_modules_blocks_modules_44_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_44_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_589 = torch.nn.functional.dropout(x_588, 0.0, False, False)
        x_588 = None
        x_590 = x_583 + x_589
        x_583 = x_589 = None
        x_591 = torch.nn.functional.layer_norm(
            x_590,
            (1664,),
            l_self_modules_blocks_modules_45_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_45_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_45_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_45_modules_norm1_parameters_bias_
        ) = None
        linear_180 = torch._C._nn.linear(
            x_591,
            l_self_modules_blocks_modules_45_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_45_modules_attn_modules_qkv_parameters_bias_,
        )
        x_591 = (
            l_self_modules_blocks_modules_45_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_45_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_90 = linear_180.reshape(1, 257, 3, 16, 104)
        linear_180 = None
        qkv_45 = reshape_90.permute(2, 0, 3, 1, 4)
        reshape_90 = None
        unbind_45 = qkv_45.unbind(0)
        qkv_45 = None
        q_45 = unbind_45[0]
        k_45 = unbind_45[1]
        v_45 = unbind_45[2]
        unbind_45 = None
        x_592 = torch._C._nn.scaled_dot_product_attention(
            q_45, k_45, v_45, attn_mask=None, dropout_p=0.0
        )
        q_45 = k_45 = v_45 = None
        transpose_46 = x_592.transpose(1, 2)
        x_592 = None
        x_593 = transpose_46.reshape(1, 257, 1664)
        transpose_46 = None
        x_594 = torch._C._nn.linear(
            x_593,
            l_self_modules_blocks_modules_45_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_45_modules_attn_modules_proj_parameters_bias_,
        )
        x_593 = l_self_modules_blocks_modules_45_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_45_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_595 = torch.nn.functional.dropout(x_594, 0.0, False, False)
        x_594 = None
        x_596 = x_590 + x_595
        x_590 = x_595 = None
        x_597 = torch.nn.functional.layer_norm(
            x_596,
            (1664,),
            l_self_modules_blocks_modules_45_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_45_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_45_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_45_modules_norm2_parameters_bias_
        ) = None
        x_598 = torch._C._nn.linear(
            x_597,
            l_self_modules_blocks_modules_45_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_45_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_597 = (
            l_self_modules_blocks_modules_45_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_45_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_90 = 1.702 * x_598
        sigmoid_45 = torch.sigmoid(mul_90)
        mul_90 = None
        x_599 = x_598 * sigmoid_45
        x_598 = sigmoid_45 = None
        x_600 = torch.nn.functional.dropout(x_599, 0.0, False, False)
        x_599 = None
        x_601 = torch._C._nn.linear(
            x_600,
            l_self_modules_blocks_modules_45_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_45_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_600 = (
            l_self_modules_blocks_modules_45_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_45_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_602 = torch.nn.functional.dropout(x_601, 0.0, False, False)
        x_601 = None
        x_603 = x_596 + x_602
        x_596 = x_602 = None
        x_604 = torch.nn.functional.layer_norm(
            x_603,
            (1664,),
            l_self_modules_blocks_modules_46_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_46_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_46_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_46_modules_norm1_parameters_bias_
        ) = None
        linear_184 = torch._C._nn.linear(
            x_604,
            l_self_modules_blocks_modules_46_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_46_modules_attn_modules_qkv_parameters_bias_,
        )
        x_604 = (
            l_self_modules_blocks_modules_46_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_46_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_92 = linear_184.reshape(1, 257, 3, 16, 104)
        linear_184 = None
        qkv_46 = reshape_92.permute(2, 0, 3, 1, 4)
        reshape_92 = None
        unbind_46 = qkv_46.unbind(0)
        qkv_46 = None
        q_46 = unbind_46[0]
        k_46 = unbind_46[1]
        v_46 = unbind_46[2]
        unbind_46 = None
        x_605 = torch._C._nn.scaled_dot_product_attention(
            q_46, k_46, v_46, attn_mask=None, dropout_p=0.0
        )
        q_46 = k_46 = v_46 = None
        transpose_47 = x_605.transpose(1, 2)
        x_605 = None
        x_606 = transpose_47.reshape(1, 257, 1664)
        transpose_47 = None
        x_607 = torch._C._nn.linear(
            x_606,
            l_self_modules_blocks_modules_46_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_46_modules_attn_modules_proj_parameters_bias_,
        )
        x_606 = l_self_modules_blocks_modules_46_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_46_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_608 = torch.nn.functional.dropout(x_607, 0.0, False, False)
        x_607 = None
        x_609 = x_603 + x_608
        x_603 = x_608 = None
        x_610 = torch.nn.functional.layer_norm(
            x_609,
            (1664,),
            l_self_modules_blocks_modules_46_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_46_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_46_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_46_modules_norm2_parameters_bias_
        ) = None
        x_611 = torch._C._nn.linear(
            x_610,
            l_self_modules_blocks_modules_46_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_46_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_610 = (
            l_self_modules_blocks_modules_46_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_46_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_92 = 1.702 * x_611
        sigmoid_46 = torch.sigmoid(mul_92)
        mul_92 = None
        x_612 = x_611 * sigmoid_46
        x_611 = sigmoid_46 = None
        x_613 = torch.nn.functional.dropout(x_612, 0.0, False, False)
        x_612 = None
        x_614 = torch._C._nn.linear(
            x_613,
            l_self_modules_blocks_modules_46_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_46_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_613 = (
            l_self_modules_blocks_modules_46_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_46_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_615 = torch.nn.functional.dropout(x_614, 0.0, False, False)
        x_614 = None
        x_616 = x_609 + x_615
        x_609 = x_615 = None
        x_617 = torch.nn.functional.layer_norm(
            x_616,
            (1664,),
            l_self_modules_blocks_modules_47_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_47_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_47_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_47_modules_norm1_parameters_bias_
        ) = None
        linear_188 = torch._C._nn.linear(
            x_617,
            l_self_modules_blocks_modules_47_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_47_modules_attn_modules_qkv_parameters_bias_,
        )
        x_617 = (
            l_self_modules_blocks_modules_47_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_47_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_94 = linear_188.reshape(1, 257, 3, 16, 104)
        linear_188 = None
        qkv_47 = reshape_94.permute(2, 0, 3, 1, 4)
        reshape_94 = None
        unbind_47 = qkv_47.unbind(0)
        qkv_47 = None
        q_47 = unbind_47[0]
        k_47 = unbind_47[1]
        v_47 = unbind_47[2]
        unbind_47 = None
        x_618 = torch._C._nn.scaled_dot_product_attention(
            q_47, k_47, v_47, attn_mask=None, dropout_p=0.0
        )
        q_47 = k_47 = v_47 = None
        transpose_48 = x_618.transpose(1, 2)
        x_618 = None
        x_619 = transpose_48.reshape(1, 257, 1664)
        transpose_48 = None
        x_620 = torch._C._nn.linear(
            x_619,
            l_self_modules_blocks_modules_47_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_47_modules_attn_modules_proj_parameters_bias_,
        )
        x_619 = l_self_modules_blocks_modules_47_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_47_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_621 = torch.nn.functional.dropout(x_620, 0.0, False, False)
        x_620 = None
        x_622 = x_616 + x_621
        x_616 = x_621 = None
        x_623 = torch.nn.functional.layer_norm(
            x_622,
            (1664,),
            l_self_modules_blocks_modules_47_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_47_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_47_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_47_modules_norm2_parameters_bias_
        ) = None
        x_624 = torch._C._nn.linear(
            x_623,
            l_self_modules_blocks_modules_47_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_47_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_623 = (
            l_self_modules_blocks_modules_47_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_47_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        mul_94 = 1.702 * x_624
        sigmoid_47 = torch.sigmoid(mul_94)
        mul_94 = None
        x_625 = x_624 * sigmoid_47
        x_624 = sigmoid_47 = None
        x_626 = torch.nn.functional.dropout(x_625, 0.0, False, False)
        x_625 = None
        x_627 = torch._C._nn.linear(
            x_626,
            l_self_modules_blocks_modules_47_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_47_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_626 = (
            l_self_modules_blocks_modules_47_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_47_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_628 = torch.nn.functional.dropout(x_627, 0.0, False, False)
        x_627 = None
        x_629 = x_622 + x_628
        x_622 = x_628 = None
        x_630 = torch.nn.functional.layer_norm(
            x_629,
            (1664,),
            l_self_modules_norm_parameters_weight_,
            l_self_modules_norm_parameters_bias_,
            1e-05,
        )
        x_629 = (
            l_self_modules_norm_parameters_weight_
        ) = l_self_modules_norm_parameters_bias_ = None
        x_631 = x_630[(slice(None, None, None), 0)]
        x_630 = None
        x_632 = torch.nn.functional.dropout(x_631, 0.0, False, False)
        x_631 = None
        x_633 = torch._C._nn.linear(
            x_632,
            l_self_modules_head_parameters_weight_,
            l_self_modules_head_parameters_bias_,
        )
        x_632 = (
            l_self_modules_head_parameters_weight_
        ) = l_self_modules_head_parameters_bias_ = None
        return (x_633,)
