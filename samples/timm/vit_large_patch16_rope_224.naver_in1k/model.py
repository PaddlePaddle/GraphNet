import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_x_: torch.Tensor,
        L_self_modules_patch_embed_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_patch_embed_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_rope_buffers_pos_embed_: torch.Tensor,
        L_self_parameters_cls_token_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_x_ = L_x_
        l_self_modules_patch_embed_modules_proj_parameters_weight_ = (
            L_self_modules_patch_embed_modules_proj_parameters_weight_
        )
        l_self_modules_patch_embed_modules_proj_parameters_bias_ = (
            L_self_modules_patch_embed_modules_proj_parameters_bias_
        )
        l_self_modules_rope_buffers_pos_embed_ = L_self_modules_rope_buffers_pos_embed_
        l_self_parameters_cls_token_ = L_self_parameters_cls_token_
        l_self_modules_blocks_modules_0_parameters_gamma_1_ = (
            L_self_modules_blocks_modules_0_parameters_gamma_1_
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
        l_self_modules_blocks_modules_0_parameters_gamma_2_ = (
            L_self_modules_blocks_modules_0_parameters_gamma_2_
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
        l_self_modules_blocks_modules_1_parameters_gamma_1_ = (
            L_self_modules_blocks_modules_1_parameters_gamma_1_
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
        l_self_modules_blocks_modules_1_parameters_gamma_2_ = (
            L_self_modules_blocks_modules_1_parameters_gamma_2_
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
        l_self_modules_blocks_modules_2_parameters_gamma_1_ = (
            L_self_modules_blocks_modules_2_parameters_gamma_1_
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
        l_self_modules_blocks_modules_2_parameters_gamma_2_ = (
            L_self_modules_blocks_modules_2_parameters_gamma_2_
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
        l_self_modules_blocks_modules_3_parameters_gamma_1_ = (
            L_self_modules_blocks_modules_3_parameters_gamma_1_
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
        l_self_modules_blocks_modules_3_parameters_gamma_2_ = (
            L_self_modules_blocks_modules_3_parameters_gamma_2_
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
        l_self_modules_blocks_modules_4_parameters_gamma_1_ = (
            L_self_modules_blocks_modules_4_parameters_gamma_1_
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
        l_self_modules_blocks_modules_4_parameters_gamma_2_ = (
            L_self_modules_blocks_modules_4_parameters_gamma_2_
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
        l_self_modules_blocks_modules_5_parameters_gamma_1_ = (
            L_self_modules_blocks_modules_5_parameters_gamma_1_
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
        l_self_modules_blocks_modules_5_parameters_gamma_2_ = (
            L_self_modules_blocks_modules_5_parameters_gamma_2_
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
        l_self_modules_blocks_modules_6_parameters_gamma_1_ = (
            L_self_modules_blocks_modules_6_parameters_gamma_1_
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
        l_self_modules_blocks_modules_6_parameters_gamma_2_ = (
            L_self_modules_blocks_modules_6_parameters_gamma_2_
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
        l_self_modules_blocks_modules_7_parameters_gamma_1_ = (
            L_self_modules_blocks_modules_7_parameters_gamma_1_
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
        l_self_modules_blocks_modules_7_parameters_gamma_2_ = (
            L_self_modules_blocks_modules_7_parameters_gamma_2_
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
        l_self_modules_blocks_modules_8_parameters_gamma_1_ = (
            L_self_modules_blocks_modules_8_parameters_gamma_1_
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
        l_self_modules_blocks_modules_8_parameters_gamma_2_ = (
            L_self_modules_blocks_modules_8_parameters_gamma_2_
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
        l_self_modules_blocks_modules_9_parameters_gamma_1_ = (
            L_self_modules_blocks_modules_9_parameters_gamma_1_
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
        l_self_modules_blocks_modules_9_parameters_gamma_2_ = (
            L_self_modules_blocks_modules_9_parameters_gamma_2_
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
        l_self_modules_blocks_modules_10_parameters_gamma_1_ = (
            L_self_modules_blocks_modules_10_parameters_gamma_1_
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
        l_self_modules_blocks_modules_10_parameters_gamma_2_ = (
            L_self_modules_blocks_modules_10_parameters_gamma_2_
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
        l_self_modules_blocks_modules_11_parameters_gamma_1_ = (
            L_self_modules_blocks_modules_11_parameters_gamma_1_
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
        l_self_modules_blocks_modules_11_parameters_gamma_2_ = (
            L_self_modules_blocks_modules_11_parameters_gamma_2_
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
        l_self_modules_blocks_modules_12_parameters_gamma_1_ = (
            L_self_modules_blocks_modules_12_parameters_gamma_1_
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
        l_self_modules_blocks_modules_12_parameters_gamma_2_ = (
            L_self_modules_blocks_modules_12_parameters_gamma_2_
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
        l_self_modules_blocks_modules_13_parameters_gamma_1_ = (
            L_self_modules_blocks_modules_13_parameters_gamma_1_
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
        l_self_modules_blocks_modules_13_parameters_gamma_2_ = (
            L_self_modules_blocks_modules_13_parameters_gamma_2_
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
        l_self_modules_blocks_modules_14_parameters_gamma_1_ = (
            L_self_modules_blocks_modules_14_parameters_gamma_1_
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
        l_self_modules_blocks_modules_14_parameters_gamma_2_ = (
            L_self_modules_blocks_modules_14_parameters_gamma_2_
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
        l_self_modules_blocks_modules_15_parameters_gamma_1_ = (
            L_self_modules_blocks_modules_15_parameters_gamma_1_
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
        l_self_modules_blocks_modules_15_parameters_gamma_2_ = (
            L_self_modules_blocks_modules_15_parameters_gamma_2_
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
        l_self_modules_blocks_modules_16_parameters_gamma_1_ = (
            L_self_modules_blocks_modules_16_parameters_gamma_1_
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
        l_self_modules_blocks_modules_16_parameters_gamma_2_ = (
            L_self_modules_blocks_modules_16_parameters_gamma_2_
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
        l_self_modules_blocks_modules_17_parameters_gamma_1_ = (
            L_self_modules_blocks_modules_17_parameters_gamma_1_
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
        l_self_modules_blocks_modules_17_parameters_gamma_2_ = (
            L_self_modules_blocks_modules_17_parameters_gamma_2_
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
        l_self_modules_blocks_modules_18_parameters_gamma_1_ = (
            L_self_modules_blocks_modules_18_parameters_gamma_1_
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
        l_self_modules_blocks_modules_18_parameters_gamma_2_ = (
            L_self_modules_blocks_modules_18_parameters_gamma_2_
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
        l_self_modules_blocks_modules_19_parameters_gamma_1_ = (
            L_self_modules_blocks_modules_19_parameters_gamma_1_
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
        l_self_modules_blocks_modules_19_parameters_gamma_2_ = (
            L_self_modules_blocks_modules_19_parameters_gamma_2_
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
        l_self_modules_blocks_modules_20_parameters_gamma_1_ = (
            L_self_modules_blocks_modules_20_parameters_gamma_1_
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
        l_self_modules_blocks_modules_20_parameters_gamma_2_ = (
            L_self_modules_blocks_modules_20_parameters_gamma_2_
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
        l_self_modules_blocks_modules_21_parameters_gamma_1_ = (
            L_self_modules_blocks_modules_21_parameters_gamma_1_
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
        l_self_modules_blocks_modules_21_parameters_gamma_2_ = (
            L_self_modules_blocks_modules_21_parameters_gamma_2_
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
        l_self_modules_blocks_modules_22_parameters_gamma_1_ = (
            L_self_modules_blocks_modules_22_parameters_gamma_1_
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
        l_self_modules_blocks_modules_22_parameters_gamma_2_ = (
            L_self_modules_blocks_modules_22_parameters_gamma_2_
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
        l_self_modules_blocks_modules_23_parameters_gamma_1_ = (
            L_self_modules_blocks_modules_23_parameters_gamma_1_
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
        l_self_modules_blocks_modules_23_parameters_gamma_2_ = (
            L_self_modules_blocks_modules_23_parameters_gamma_2_
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
        l_self_modules_norm_parameters_weight_ = L_self_modules_norm_parameters_weight_
        l_self_modules_norm_parameters_bias_ = L_self_modules_norm_parameters_bias_
        l_self_modules_head_parameters_weight_ = L_self_modules_head_parameters_weight_
        l_self_modules_head_parameters_bias_ = L_self_modules_head_parameters_bias_
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
        expand = l_self_parameters_cls_token_.expand(1, -1, -1)
        l_self_parameters_cls_token_ = None
        x_2 = torch.cat([expand, x_1], dim=1)
        expand = x_1 = None
        x_3 = torch.nn.functional.dropout(x_2, 0.0, False, False)
        x_2 = None
        x_4 = torch.nn.functional.layer_norm(
            x_3,
            (1024,),
            l_self_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_0_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_0_modules_norm1_parameters_bias_
        ) = None
        qkv = torch._C._nn.linear(
            x_4,
            l_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_,
        )
        x_4 = (
            l_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape = qkv.reshape(1, 197, 3, 16, -1)
        qkv = None
        qkv_1 = reshape.permute(2, 0, 3, 1, 4)
        reshape = None
        unbind = qkv_1.unbind(0)
        qkv_1 = None
        q = unbind[0]
        k = unbind[1]
        v = unbind[2]
        unbind = None
        getitem_7 = q[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_8 = q[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q = None
        tensor_split = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb = tensor_split[0]
        cos_emb = tensor_split[1]
        tensor_split = None
        mul = getitem_8 * cos_emb
        cos_emb = None
        getitem_11 = getitem_8[(Ellipsis, slice(1, None, 2))]
        neg = -getitem_11
        getitem_11 = None
        getitem_12 = getitem_8[(Ellipsis, slice(None, None, 2))]
        getitem_8 = None
        stack = torch.stack([neg, getitem_12], -1)
        neg = getitem_12 = None
        reshape_1 = stack.reshape((1, 16, 196, 64))
        stack = None
        mul_1 = reshape_1 * sin_emb
        reshape_1 = sin_emb = None
        add = mul + mul_1
        mul = mul_1 = None
        cat_1 = torch.cat([getitem_7, add], dim=2)
        getitem_7 = add = None
        q_1 = cat_1.type_as(v)
        cat_1 = None
        getitem_13 = k[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_14 = k[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k = None
        tensor_split_1 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_1 = tensor_split_1[0]
        cos_emb_1 = tensor_split_1[1]
        tensor_split_1 = None
        mul_2 = getitem_14 * cos_emb_1
        cos_emb_1 = None
        getitem_17 = getitem_14[(Ellipsis, slice(1, None, 2))]
        neg_1 = -getitem_17
        getitem_17 = None
        getitem_18 = getitem_14[(Ellipsis, slice(None, None, 2))]
        getitem_14 = None
        stack_1 = torch.stack([neg_1, getitem_18], -1)
        neg_1 = getitem_18 = None
        reshape_2 = stack_1.reshape((1, 16, 196, 64))
        stack_1 = None
        mul_3 = reshape_2 * sin_emb_1
        reshape_2 = sin_emb_1 = None
        add_1 = mul_2 + mul_3
        mul_2 = mul_3 = None
        cat_2 = torch.cat([getitem_13, add_1], dim=2)
        getitem_13 = add_1 = None
        k_1 = cat_2.type_as(v)
        cat_2 = None
        x_5 = torch._C._nn.scaled_dot_product_attention(
            q_1, k_1, v, attn_mask=None, dropout_p=0.0
        )
        q_1 = k_1 = v = None
        transpose_1 = x_5.transpose(1, 2)
        x_5 = None
        x_6 = transpose_1.reshape(1, 197, 1024)
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
        mul_4 = l_self_modules_blocks_modules_0_parameters_gamma_1_ * x_8
        l_self_modules_blocks_modules_0_parameters_gamma_1_ = x_8 = None
        x_9 = x_3 + mul_4
        x_3 = mul_4 = None
        x_10 = torch.nn.functional.layer_norm(
            x_9,
            (1024,),
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
        mul_5 = l_self_modules_blocks_modules_0_parameters_gamma_2_ * x_15
        l_self_modules_blocks_modules_0_parameters_gamma_2_ = x_15 = None
        x_16 = x_9 + mul_5
        x_9 = mul_5 = None
        x_17 = torch.nn.functional.layer_norm(
            x_16,
            (1024,),
            l_self_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_1_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_1_modules_norm1_parameters_bias_
        ) = None
        qkv_2 = torch._C._nn.linear(
            x_17,
            l_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_,
        )
        x_17 = (
            l_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_4 = qkv_2.reshape(1, 197, 3, 16, -1)
        qkv_2 = None
        qkv_3 = reshape_4.permute(2, 0, 3, 1, 4)
        reshape_4 = None
        unbind_1 = qkv_3.unbind(0)
        qkv_3 = None
        q_2 = unbind_1[0]
        k_2 = unbind_1[1]
        v_1 = unbind_1[2]
        unbind_1 = None
        getitem_22 = q_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_23 = q_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_2 = None
        tensor_split_2 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_2 = tensor_split_2[0]
        cos_emb_2 = tensor_split_2[1]
        tensor_split_2 = None
        mul_6 = getitem_23 * cos_emb_2
        cos_emb_2 = None
        getitem_26 = getitem_23[(Ellipsis, slice(1, None, 2))]
        neg_2 = -getitem_26
        getitem_26 = None
        getitem_27 = getitem_23[(Ellipsis, slice(None, None, 2))]
        getitem_23 = None
        stack_2 = torch.stack([neg_2, getitem_27], -1)
        neg_2 = getitem_27 = None
        reshape_5 = stack_2.reshape((1, 16, 196, 64))
        stack_2 = None
        mul_7 = reshape_5 * sin_emb_2
        reshape_5 = sin_emb_2 = None
        add_4 = mul_6 + mul_7
        mul_6 = mul_7 = None
        cat_3 = torch.cat([getitem_22, add_4], dim=2)
        getitem_22 = add_4 = None
        q_3 = cat_3.type_as(v_1)
        cat_3 = None
        getitem_28 = k_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_29 = k_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_2 = None
        tensor_split_3 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_3 = tensor_split_3[0]
        cos_emb_3 = tensor_split_3[1]
        tensor_split_3 = None
        mul_8 = getitem_29 * cos_emb_3
        cos_emb_3 = None
        getitem_32 = getitem_29[(Ellipsis, slice(1, None, 2))]
        neg_3 = -getitem_32
        getitem_32 = None
        getitem_33 = getitem_29[(Ellipsis, slice(None, None, 2))]
        getitem_29 = None
        stack_3 = torch.stack([neg_3, getitem_33], -1)
        neg_3 = getitem_33 = None
        reshape_6 = stack_3.reshape((1, 16, 196, 64))
        stack_3 = None
        mul_9 = reshape_6 * sin_emb_3
        reshape_6 = sin_emb_3 = None
        add_5 = mul_8 + mul_9
        mul_8 = mul_9 = None
        cat_4 = torch.cat([getitem_28, add_5], dim=2)
        getitem_28 = add_5 = None
        k_3 = cat_4.type_as(v_1)
        cat_4 = None
        x_18 = torch._C._nn.scaled_dot_product_attention(
            q_3, k_3, v_1, attn_mask=None, dropout_p=0.0
        )
        q_3 = k_3 = v_1 = None
        transpose_2 = x_18.transpose(1, 2)
        x_18 = None
        x_19 = transpose_2.reshape(1, 197, 1024)
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
        mul_10 = l_self_modules_blocks_modules_1_parameters_gamma_1_ * x_21
        l_self_modules_blocks_modules_1_parameters_gamma_1_ = x_21 = None
        x_22 = x_16 + mul_10
        x_16 = mul_10 = None
        x_23 = torch.nn.functional.layer_norm(
            x_22,
            (1024,),
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
        mul_11 = l_self_modules_blocks_modules_1_parameters_gamma_2_ * x_28
        l_self_modules_blocks_modules_1_parameters_gamma_2_ = x_28 = None
        x_29 = x_22 + mul_11
        x_22 = mul_11 = None
        x_30 = torch.nn.functional.layer_norm(
            x_29,
            (1024,),
            l_self_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_2_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_2_modules_norm1_parameters_bias_
        ) = None
        qkv_4 = torch._C._nn.linear(
            x_30,
            l_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_,
        )
        x_30 = (
            l_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_8 = qkv_4.reshape(1, 197, 3, 16, -1)
        qkv_4 = None
        qkv_5 = reshape_8.permute(2, 0, 3, 1, 4)
        reshape_8 = None
        unbind_2 = qkv_5.unbind(0)
        qkv_5 = None
        q_4 = unbind_2[0]
        k_4 = unbind_2[1]
        v_2 = unbind_2[2]
        unbind_2 = None
        getitem_37 = q_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_38 = q_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_4 = None
        tensor_split_4 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_4 = tensor_split_4[0]
        cos_emb_4 = tensor_split_4[1]
        tensor_split_4 = None
        mul_12 = getitem_38 * cos_emb_4
        cos_emb_4 = None
        getitem_41 = getitem_38[(Ellipsis, slice(1, None, 2))]
        neg_4 = -getitem_41
        getitem_41 = None
        getitem_42 = getitem_38[(Ellipsis, slice(None, None, 2))]
        getitem_38 = None
        stack_4 = torch.stack([neg_4, getitem_42], -1)
        neg_4 = getitem_42 = None
        reshape_9 = stack_4.reshape((1, 16, 196, 64))
        stack_4 = None
        mul_13 = reshape_9 * sin_emb_4
        reshape_9 = sin_emb_4 = None
        add_8 = mul_12 + mul_13
        mul_12 = mul_13 = None
        cat_5 = torch.cat([getitem_37, add_8], dim=2)
        getitem_37 = add_8 = None
        q_5 = cat_5.type_as(v_2)
        cat_5 = None
        getitem_43 = k_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_44 = k_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_4 = None
        tensor_split_5 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_5 = tensor_split_5[0]
        cos_emb_5 = tensor_split_5[1]
        tensor_split_5 = None
        mul_14 = getitem_44 * cos_emb_5
        cos_emb_5 = None
        getitem_47 = getitem_44[(Ellipsis, slice(1, None, 2))]
        neg_5 = -getitem_47
        getitem_47 = None
        getitem_48 = getitem_44[(Ellipsis, slice(None, None, 2))]
        getitem_44 = None
        stack_5 = torch.stack([neg_5, getitem_48], -1)
        neg_5 = getitem_48 = None
        reshape_10 = stack_5.reshape((1, 16, 196, 64))
        stack_5 = None
        mul_15 = reshape_10 * sin_emb_5
        reshape_10 = sin_emb_5 = None
        add_9 = mul_14 + mul_15
        mul_14 = mul_15 = None
        cat_6 = torch.cat([getitem_43, add_9], dim=2)
        getitem_43 = add_9 = None
        k_5 = cat_6.type_as(v_2)
        cat_6 = None
        x_31 = torch._C._nn.scaled_dot_product_attention(
            q_5, k_5, v_2, attn_mask=None, dropout_p=0.0
        )
        q_5 = k_5 = v_2 = None
        transpose_3 = x_31.transpose(1, 2)
        x_31 = None
        x_32 = transpose_3.reshape(1, 197, 1024)
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
        mul_16 = l_self_modules_blocks_modules_2_parameters_gamma_1_ * x_34
        l_self_modules_blocks_modules_2_parameters_gamma_1_ = x_34 = None
        x_35 = x_29 + mul_16
        x_29 = mul_16 = None
        x_36 = torch.nn.functional.layer_norm(
            x_35,
            (1024,),
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
        mul_17 = l_self_modules_blocks_modules_2_parameters_gamma_2_ * x_41
        l_self_modules_blocks_modules_2_parameters_gamma_2_ = x_41 = None
        x_42 = x_35 + mul_17
        x_35 = mul_17 = None
        x_43 = torch.nn.functional.layer_norm(
            x_42,
            (1024,),
            l_self_modules_blocks_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_3_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_3_modules_norm1_parameters_bias_
        ) = None
        qkv_6 = torch._C._nn.linear(
            x_43,
            l_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_bias_,
        )
        x_43 = (
            l_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_12 = qkv_6.reshape(1, 197, 3, 16, -1)
        qkv_6 = None
        qkv_7 = reshape_12.permute(2, 0, 3, 1, 4)
        reshape_12 = None
        unbind_3 = qkv_7.unbind(0)
        qkv_7 = None
        q_6 = unbind_3[0]
        k_6 = unbind_3[1]
        v_3 = unbind_3[2]
        unbind_3 = None
        getitem_52 = q_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_53 = q_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_6 = None
        tensor_split_6 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_6 = tensor_split_6[0]
        cos_emb_6 = tensor_split_6[1]
        tensor_split_6 = None
        mul_18 = getitem_53 * cos_emb_6
        cos_emb_6 = None
        getitem_56 = getitem_53[(Ellipsis, slice(1, None, 2))]
        neg_6 = -getitem_56
        getitem_56 = None
        getitem_57 = getitem_53[(Ellipsis, slice(None, None, 2))]
        getitem_53 = None
        stack_6 = torch.stack([neg_6, getitem_57], -1)
        neg_6 = getitem_57 = None
        reshape_13 = stack_6.reshape((1, 16, 196, 64))
        stack_6 = None
        mul_19 = reshape_13 * sin_emb_6
        reshape_13 = sin_emb_6 = None
        add_12 = mul_18 + mul_19
        mul_18 = mul_19 = None
        cat_7 = torch.cat([getitem_52, add_12], dim=2)
        getitem_52 = add_12 = None
        q_7 = cat_7.type_as(v_3)
        cat_7 = None
        getitem_58 = k_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_59 = k_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_6 = None
        tensor_split_7 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_7 = tensor_split_7[0]
        cos_emb_7 = tensor_split_7[1]
        tensor_split_7 = None
        mul_20 = getitem_59 * cos_emb_7
        cos_emb_7 = None
        getitem_62 = getitem_59[(Ellipsis, slice(1, None, 2))]
        neg_7 = -getitem_62
        getitem_62 = None
        getitem_63 = getitem_59[(Ellipsis, slice(None, None, 2))]
        getitem_59 = None
        stack_7 = torch.stack([neg_7, getitem_63], -1)
        neg_7 = getitem_63 = None
        reshape_14 = stack_7.reshape((1, 16, 196, 64))
        stack_7 = None
        mul_21 = reshape_14 * sin_emb_7
        reshape_14 = sin_emb_7 = None
        add_13 = mul_20 + mul_21
        mul_20 = mul_21 = None
        cat_8 = torch.cat([getitem_58, add_13], dim=2)
        getitem_58 = add_13 = None
        k_7 = cat_8.type_as(v_3)
        cat_8 = None
        x_44 = torch._C._nn.scaled_dot_product_attention(
            q_7, k_7, v_3, attn_mask=None, dropout_p=0.0
        )
        q_7 = k_7 = v_3 = None
        transpose_4 = x_44.transpose(1, 2)
        x_44 = None
        x_45 = transpose_4.reshape(1, 197, 1024)
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
        mul_22 = l_self_modules_blocks_modules_3_parameters_gamma_1_ * x_47
        l_self_modules_blocks_modules_3_parameters_gamma_1_ = x_47 = None
        x_48 = x_42 + mul_22
        x_42 = mul_22 = None
        x_49 = torch.nn.functional.layer_norm(
            x_48,
            (1024,),
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
        mul_23 = l_self_modules_blocks_modules_3_parameters_gamma_2_ * x_54
        l_self_modules_blocks_modules_3_parameters_gamma_2_ = x_54 = None
        x_55 = x_48 + mul_23
        x_48 = mul_23 = None
        x_56 = torch.nn.functional.layer_norm(
            x_55,
            (1024,),
            l_self_modules_blocks_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_4_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_4_modules_norm1_parameters_bias_
        ) = None
        qkv_8 = torch._C._nn.linear(
            x_56,
            l_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_bias_,
        )
        x_56 = (
            l_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_16 = qkv_8.reshape(1, 197, 3, 16, -1)
        qkv_8 = None
        qkv_9 = reshape_16.permute(2, 0, 3, 1, 4)
        reshape_16 = None
        unbind_4 = qkv_9.unbind(0)
        qkv_9 = None
        q_8 = unbind_4[0]
        k_8 = unbind_4[1]
        v_4 = unbind_4[2]
        unbind_4 = None
        getitem_67 = q_8[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_68 = q_8[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_8 = None
        tensor_split_8 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_8 = tensor_split_8[0]
        cos_emb_8 = tensor_split_8[1]
        tensor_split_8 = None
        mul_24 = getitem_68 * cos_emb_8
        cos_emb_8 = None
        getitem_71 = getitem_68[(Ellipsis, slice(1, None, 2))]
        neg_8 = -getitem_71
        getitem_71 = None
        getitem_72 = getitem_68[(Ellipsis, slice(None, None, 2))]
        getitem_68 = None
        stack_8 = torch.stack([neg_8, getitem_72], -1)
        neg_8 = getitem_72 = None
        reshape_17 = stack_8.reshape((1, 16, 196, 64))
        stack_8 = None
        mul_25 = reshape_17 * sin_emb_8
        reshape_17 = sin_emb_8 = None
        add_16 = mul_24 + mul_25
        mul_24 = mul_25 = None
        cat_9 = torch.cat([getitem_67, add_16], dim=2)
        getitem_67 = add_16 = None
        q_9 = cat_9.type_as(v_4)
        cat_9 = None
        getitem_73 = k_8[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_74 = k_8[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_8 = None
        tensor_split_9 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_9 = tensor_split_9[0]
        cos_emb_9 = tensor_split_9[1]
        tensor_split_9 = None
        mul_26 = getitem_74 * cos_emb_9
        cos_emb_9 = None
        getitem_77 = getitem_74[(Ellipsis, slice(1, None, 2))]
        neg_9 = -getitem_77
        getitem_77 = None
        getitem_78 = getitem_74[(Ellipsis, slice(None, None, 2))]
        getitem_74 = None
        stack_9 = torch.stack([neg_9, getitem_78], -1)
        neg_9 = getitem_78 = None
        reshape_18 = stack_9.reshape((1, 16, 196, 64))
        stack_9 = None
        mul_27 = reshape_18 * sin_emb_9
        reshape_18 = sin_emb_9 = None
        add_17 = mul_26 + mul_27
        mul_26 = mul_27 = None
        cat_10 = torch.cat([getitem_73, add_17], dim=2)
        getitem_73 = add_17 = None
        k_9 = cat_10.type_as(v_4)
        cat_10 = None
        x_57 = torch._C._nn.scaled_dot_product_attention(
            q_9, k_9, v_4, attn_mask=None, dropout_p=0.0
        )
        q_9 = k_9 = v_4 = None
        transpose_5 = x_57.transpose(1, 2)
        x_57 = None
        x_58 = transpose_5.reshape(1, 197, 1024)
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
        mul_28 = l_self_modules_blocks_modules_4_parameters_gamma_1_ * x_60
        l_self_modules_blocks_modules_4_parameters_gamma_1_ = x_60 = None
        x_61 = x_55 + mul_28
        x_55 = mul_28 = None
        x_62 = torch.nn.functional.layer_norm(
            x_61,
            (1024,),
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
        mul_29 = l_self_modules_blocks_modules_4_parameters_gamma_2_ * x_67
        l_self_modules_blocks_modules_4_parameters_gamma_2_ = x_67 = None
        x_68 = x_61 + mul_29
        x_61 = mul_29 = None
        x_69 = torch.nn.functional.layer_norm(
            x_68,
            (1024,),
            l_self_modules_blocks_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_5_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_5_modules_norm1_parameters_bias_
        ) = None
        qkv_10 = torch._C._nn.linear(
            x_69,
            l_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_bias_,
        )
        x_69 = (
            l_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_20 = qkv_10.reshape(1, 197, 3, 16, -1)
        qkv_10 = None
        qkv_11 = reshape_20.permute(2, 0, 3, 1, 4)
        reshape_20 = None
        unbind_5 = qkv_11.unbind(0)
        qkv_11 = None
        q_10 = unbind_5[0]
        k_10 = unbind_5[1]
        v_5 = unbind_5[2]
        unbind_5 = None
        getitem_82 = q_10[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_83 = q_10[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_10 = None
        tensor_split_10 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_10 = tensor_split_10[0]
        cos_emb_10 = tensor_split_10[1]
        tensor_split_10 = None
        mul_30 = getitem_83 * cos_emb_10
        cos_emb_10 = None
        getitem_86 = getitem_83[(Ellipsis, slice(1, None, 2))]
        neg_10 = -getitem_86
        getitem_86 = None
        getitem_87 = getitem_83[(Ellipsis, slice(None, None, 2))]
        getitem_83 = None
        stack_10 = torch.stack([neg_10, getitem_87], -1)
        neg_10 = getitem_87 = None
        reshape_21 = stack_10.reshape((1, 16, 196, 64))
        stack_10 = None
        mul_31 = reshape_21 * sin_emb_10
        reshape_21 = sin_emb_10 = None
        add_20 = mul_30 + mul_31
        mul_30 = mul_31 = None
        cat_11 = torch.cat([getitem_82, add_20], dim=2)
        getitem_82 = add_20 = None
        q_11 = cat_11.type_as(v_5)
        cat_11 = None
        getitem_88 = k_10[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_89 = k_10[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_10 = None
        tensor_split_11 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_11 = tensor_split_11[0]
        cos_emb_11 = tensor_split_11[1]
        tensor_split_11 = None
        mul_32 = getitem_89 * cos_emb_11
        cos_emb_11 = None
        getitem_92 = getitem_89[(Ellipsis, slice(1, None, 2))]
        neg_11 = -getitem_92
        getitem_92 = None
        getitem_93 = getitem_89[(Ellipsis, slice(None, None, 2))]
        getitem_89 = None
        stack_11 = torch.stack([neg_11, getitem_93], -1)
        neg_11 = getitem_93 = None
        reshape_22 = stack_11.reshape((1, 16, 196, 64))
        stack_11 = None
        mul_33 = reshape_22 * sin_emb_11
        reshape_22 = sin_emb_11 = None
        add_21 = mul_32 + mul_33
        mul_32 = mul_33 = None
        cat_12 = torch.cat([getitem_88, add_21], dim=2)
        getitem_88 = add_21 = None
        k_11 = cat_12.type_as(v_5)
        cat_12 = None
        x_70 = torch._C._nn.scaled_dot_product_attention(
            q_11, k_11, v_5, attn_mask=None, dropout_p=0.0
        )
        q_11 = k_11 = v_5 = None
        transpose_6 = x_70.transpose(1, 2)
        x_70 = None
        x_71 = transpose_6.reshape(1, 197, 1024)
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
        mul_34 = l_self_modules_blocks_modules_5_parameters_gamma_1_ * x_73
        l_self_modules_blocks_modules_5_parameters_gamma_1_ = x_73 = None
        x_74 = x_68 + mul_34
        x_68 = mul_34 = None
        x_75 = torch.nn.functional.layer_norm(
            x_74,
            (1024,),
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
        mul_35 = l_self_modules_blocks_modules_5_parameters_gamma_2_ * x_80
        l_self_modules_blocks_modules_5_parameters_gamma_2_ = x_80 = None
        x_81 = x_74 + mul_35
        x_74 = mul_35 = None
        x_82 = torch.nn.functional.layer_norm(
            x_81,
            (1024,),
            l_self_modules_blocks_modules_6_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_6_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_6_modules_norm1_parameters_bias_
        ) = None
        qkv_12 = torch._C._nn.linear(
            x_82,
            l_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_bias_,
        )
        x_82 = (
            l_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_24 = qkv_12.reshape(1, 197, 3, 16, -1)
        qkv_12 = None
        qkv_13 = reshape_24.permute(2, 0, 3, 1, 4)
        reshape_24 = None
        unbind_6 = qkv_13.unbind(0)
        qkv_13 = None
        q_12 = unbind_6[0]
        k_12 = unbind_6[1]
        v_6 = unbind_6[2]
        unbind_6 = None
        getitem_97 = q_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_98 = q_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_12 = None
        tensor_split_12 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_12 = tensor_split_12[0]
        cos_emb_12 = tensor_split_12[1]
        tensor_split_12 = None
        mul_36 = getitem_98 * cos_emb_12
        cos_emb_12 = None
        getitem_101 = getitem_98[(Ellipsis, slice(1, None, 2))]
        neg_12 = -getitem_101
        getitem_101 = None
        getitem_102 = getitem_98[(Ellipsis, slice(None, None, 2))]
        getitem_98 = None
        stack_12 = torch.stack([neg_12, getitem_102], -1)
        neg_12 = getitem_102 = None
        reshape_25 = stack_12.reshape((1, 16, 196, 64))
        stack_12 = None
        mul_37 = reshape_25 * sin_emb_12
        reshape_25 = sin_emb_12 = None
        add_24 = mul_36 + mul_37
        mul_36 = mul_37 = None
        cat_13 = torch.cat([getitem_97, add_24], dim=2)
        getitem_97 = add_24 = None
        q_13 = cat_13.type_as(v_6)
        cat_13 = None
        getitem_103 = k_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_104 = k_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_12 = None
        tensor_split_13 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_13 = tensor_split_13[0]
        cos_emb_13 = tensor_split_13[1]
        tensor_split_13 = None
        mul_38 = getitem_104 * cos_emb_13
        cos_emb_13 = None
        getitem_107 = getitem_104[(Ellipsis, slice(1, None, 2))]
        neg_13 = -getitem_107
        getitem_107 = None
        getitem_108 = getitem_104[(Ellipsis, slice(None, None, 2))]
        getitem_104 = None
        stack_13 = torch.stack([neg_13, getitem_108], -1)
        neg_13 = getitem_108 = None
        reshape_26 = stack_13.reshape((1, 16, 196, 64))
        stack_13 = None
        mul_39 = reshape_26 * sin_emb_13
        reshape_26 = sin_emb_13 = None
        add_25 = mul_38 + mul_39
        mul_38 = mul_39 = None
        cat_14 = torch.cat([getitem_103, add_25], dim=2)
        getitem_103 = add_25 = None
        k_13 = cat_14.type_as(v_6)
        cat_14 = None
        x_83 = torch._C._nn.scaled_dot_product_attention(
            q_13, k_13, v_6, attn_mask=None, dropout_p=0.0
        )
        q_13 = k_13 = v_6 = None
        transpose_7 = x_83.transpose(1, 2)
        x_83 = None
        x_84 = transpose_7.reshape(1, 197, 1024)
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
        mul_40 = l_self_modules_blocks_modules_6_parameters_gamma_1_ * x_86
        l_self_modules_blocks_modules_6_parameters_gamma_1_ = x_86 = None
        x_87 = x_81 + mul_40
        x_81 = mul_40 = None
        x_88 = torch.nn.functional.layer_norm(
            x_87,
            (1024,),
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
        mul_41 = l_self_modules_blocks_modules_6_parameters_gamma_2_ * x_93
        l_self_modules_blocks_modules_6_parameters_gamma_2_ = x_93 = None
        x_94 = x_87 + mul_41
        x_87 = mul_41 = None
        x_95 = torch.nn.functional.layer_norm(
            x_94,
            (1024,),
            l_self_modules_blocks_modules_7_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_7_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_7_modules_norm1_parameters_bias_
        ) = None
        qkv_14 = torch._C._nn.linear(
            x_95,
            l_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_bias_,
        )
        x_95 = (
            l_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_28 = qkv_14.reshape(1, 197, 3, 16, -1)
        qkv_14 = None
        qkv_15 = reshape_28.permute(2, 0, 3, 1, 4)
        reshape_28 = None
        unbind_7 = qkv_15.unbind(0)
        qkv_15 = None
        q_14 = unbind_7[0]
        k_14 = unbind_7[1]
        v_7 = unbind_7[2]
        unbind_7 = None
        getitem_112 = q_14[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_113 = q_14[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_14 = None
        tensor_split_14 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_14 = tensor_split_14[0]
        cos_emb_14 = tensor_split_14[1]
        tensor_split_14 = None
        mul_42 = getitem_113 * cos_emb_14
        cos_emb_14 = None
        getitem_116 = getitem_113[(Ellipsis, slice(1, None, 2))]
        neg_14 = -getitem_116
        getitem_116 = None
        getitem_117 = getitem_113[(Ellipsis, slice(None, None, 2))]
        getitem_113 = None
        stack_14 = torch.stack([neg_14, getitem_117], -1)
        neg_14 = getitem_117 = None
        reshape_29 = stack_14.reshape((1, 16, 196, 64))
        stack_14 = None
        mul_43 = reshape_29 * sin_emb_14
        reshape_29 = sin_emb_14 = None
        add_28 = mul_42 + mul_43
        mul_42 = mul_43 = None
        cat_15 = torch.cat([getitem_112, add_28], dim=2)
        getitem_112 = add_28 = None
        q_15 = cat_15.type_as(v_7)
        cat_15 = None
        getitem_118 = k_14[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_119 = k_14[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_14 = None
        tensor_split_15 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_15 = tensor_split_15[0]
        cos_emb_15 = tensor_split_15[1]
        tensor_split_15 = None
        mul_44 = getitem_119 * cos_emb_15
        cos_emb_15 = None
        getitem_122 = getitem_119[(Ellipsis, slice(1, None, 2))]
        neg_15 = -getitem_122
        getitem_122 = None
        getitem_123 = getitem_119[(Ellipsis, slice(None, None, 2))]
        getitem_119 = None
        stack_15 = torch.stack([neg_15, getitem_123], -1)
        neg_15 = getitem_123 = None
        reshape_30 = stack_15.reshape((1, 16, 196, 64))
        stack_15 = None
        mul_45 = reshape_30 * sin_emb_15
        reshape_30 = sin_emb_15 = None
        add_29 = mul_44 + mul_45
        mul_44 = mul_45 = None
        cat_16 = torch.cat([getitem_118, add_29], dim=2)
        getitem_118 = add_29 = None
        k_15 = cat_16.type_as(v_7)
        cat_16 = None
        x_96 = torch._C._nn.scaled_dot_product_attention(
            q_15, k_15, v_7, attn_mask=None, dropout_p=0.0
        )
        q_15 = k_15 = v_7 = None
        transpose_8 = x_96.transpose(1, 2)
        x_96 = None
        x_97 = transpose_8.reshape(1, 197, 1024)
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
        mul_46 = l_self_modules_blocks_modules_7_parameters_gamma_1_ * x_99
        l_self_modules_blocks_modules_7_parameters_gamma_1_ = x_99 = None
        x_100 = x_94 + mul_46
        x_94 = mul_46 = None
        x_101 = torch.nn.functional.layer_norm(
            x_100,
            (1024,),
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
        mul_47 = l_self_modules_blocks_modules_7_parameters_gamma_2_ * x_106
        l_self_modules_blocks_modules_7_parameters_gamma_2_ = x_106 = None
        x_107 = x_100 + mul_47
        x_100 = mul_47 = None
        x_108 = torch.nn.functional.layer_norm(
            x_107,
            (1024,),
            l_self_modules_blocks_modules_8_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_8_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_8_modules_norm1_parameters_bias_
        ) = None
        qkv_16 = torch._C._nn.linear(
            x_108,
            l_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_bias_,
        )
        x_108 = (
            l_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_32 = qkv_16.reshape(1, 197, 3, 16, -1)
        qkv_16 = None
        qkv_17 = reshape_32.permute(2, 0, 3, 1, 4)
        reshape_32 = None
        unbind_8 = qkv_17.unbind(0)
        qkv_17 = None
        q_16 = unbind_8[0]
        k_16 = unbind_8[1]
        v_8 = unbind_8[2]
        unbind_8 = None
        getitem_127 = q_16[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_128 = q_16[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_16 = None
        tensor_split_16 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_16 = tensor_split_16[0]
        cos_emb_16 = tensor_split_16[1]
        tensor_split_16 = None
        mul_48 = getitem_128 * cos_emb_16
        cos_emb_16 = None
        getitem_131 = getitem_128[(Ellipsis, slice(1, None, 2))]
        neg_16 = -getitem_131
        getitem_131 = None
        getitem_132 = getitem_128[(Ellipsis, slice(None, None, 2))]
        getitem_128 = None
        stack_16 = torch.stack([neg_16, getitem_132], -1)
        neg_16 = getitem_132 = None
        reshape_33 = stack_16.reshape((1, 16, 196, 64))
        stack_16 = None
        mul_49 = reshape_33 * sin_emb_16
        reshape_33 = sin_emb_16 = None
        add_32 = mul_48 + mul_49
        mul_48 = mul_49 = None
        cat_17 = torch.cat([getitem_127, add_32], dim=2)
        getitem_127 = add_32 = None
        q_17 = cat_17.type_as(v_8)
        cat_17 = None
        getitem_133 = k_16[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_134 = k_16[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_16 = None
        tensor_split_17 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_17 = tensor_split_17[0]
        cos_emb_17 = tensor_split_17[1]
        tensor_split_17 = None
        mul_50 = getitem_134 * cos_emb_17
        cos_emb_17 = None
        getitem_137 = getitem_134[(Ellipsis, slice(1, None, 2))]
        neg_17 = -getitem_137
        getitem_137 = None
        getitem_138 = getitem_134[(Ellipsis, slice(None, None, 2))]
        getitem_134 = None
        stack_17 = torch.stack([neg_17, getitem_138], -1)
        neg_17 = getitem_138 = None
        reshape_34 = stack_17.reshape((1, 16, 196, 64))
        stack_17 = None
        mul_51 = reshape_34 * sin_emb_17
        reshape_34 = sin_emb_17 = None
        add_33 = mul_50 + mul_51
        mul_50 = mul_51 = None
        cat_18 = torch.cat([getitem_133, add_33], dim=2)
        getitem_133 = add_33 = None
        k_17 = cat_18.type_as(v_8)
        cat_18 = None
        x_109 = torch._C._nn.scaled_dot_product_attention(
            q_17, k_17, v_8, attn_mask=None, dropout_p=0.0
        )
        q_17 = k_17 = v_8 = None
        transpose_9 = x_109.transpose(1, 2)
        x_109 = None
        x_110 = transpose_9.reshape(1, 197, 1024)
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
        mul_52 = l_self_modules_blocks_modules_8_parameters_gamma_1_ * x_112
        l_self_modules_blocks_modules_8_parameters_gamma_1_ = x_112 = None
        x_113 = x_107 + mul_52
        x_107 = mul_52 = None
        x_114 = torch.nn.functional.layer_norm(
            x_113,
            (1024,),
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
        mul_53 = l_self_modules_blocks_modules_8_parameters_gamma_2_ * x_119
        l_self_modules_blocks_modules_8_parameters_gamma_2_ = x_119 = None
        x_120 = x_113 + mul_53
        x_113 = mul_53 = None
        x_121 = torch.nn.functional.layer_norm(
            x_120,
            (1024,),
            l_self_modules_blocks_modules_9_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_9_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_9_modules_norm1_parameters_bias_
        ) = None
        qkv_18 = torch._C._nn.linear(
            x_121,
            l_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_bias_,
        )
        x_121 = (
            l_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_36 = qkv_18.reshape(1, 197, 3, 16, -1)
        qkv_18 = None
        qkv_19 = reshape_36.permute(2, 0, 3, 1, 4)
        reshape_36 = None
        unbind_9 = qkv_19.unbind(0)
        qkv_19 = None
        q_18 = unbind_9[0]
        k_18 = unbind_9[1]
        v_9 = unbind_9[2]
        unbind_9 = None
        getitem_142 = q_18[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_143 = q_18[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_18 = None
        tensor_split_18 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_18 = tensor_split_18[0]
        cos_emb_18 = tensor_split_18[1]
        tensor_split_18 = None
        mul_54 = getitem_143 * cos_emb_18
        cos_emb_18 = None
        getitem_146 = getitem_143[(Ellipsis, slice(1, None, 2))]
        neg_18 = -getitem_146
        getitem_146 = None
        getitem_147 = getitem_143[(Ellipsis, slice(None, None, 2))]
        getitem_143 = None
        stack_18 = torch.stack([neg_18, getitem_147], -1)
        neg_18 = getitem_147 = None
        reshape_37 = stack_18.reshape((1, 16, 196, 64))
        stack_18 = None
        mul_55 = reshape_37 * sin_emb_18
        reshape_37 = sin_emb_18 = None
        add_36 = mul_54 + mul_55
        mul_54 = mul_55 = None
        cat_19 = torch.cat([getitem_142, add_36], dim=2)
        getitem_142 = add_36 = None
        q_19 = cat_19.type_as(v_9)
        cat_19 = None
        getitem_148 = k_18[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_149 = k_18[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_18 = None
        tensor_split_19 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_19 = tensor_split_19[0]
        cos_emb_19 = tensor_split_19[1]
        tensor_split_19 = None
        mul_56 = getitem_149 * cos_emb_19
        cos_emb_19 = None
        getitem_152 = getitem_149[(Ellipsis, slice(1, None, 2))]
        neg_19 = -getitem_152
        getitem_152 = None
        getitem_153 = getitem_149[(Ellipsis, slice(None, None, 2))]
        getitem_149 = None
        stack_19 = torch.stack([neg_19, getitem_153], -1)
        neg_19 = getitem_153 = None
        reshape_38 = stack_19.reshape((1, 16, 196, 64))
        stack_19 = None
        mul_57 = reshape_38 * sin_emb_19
        reshape_38 = sin_emb_19 = None
        add_37 = mul_56 + mul_57
        mul_56 = mul_57 = None
        cat_20 = torch.cat([getitem_148, add_37], dim=2)
        getitem_148 = add_37 = None
        k_19 = cat_20.type_as(v_9)
        cat_20 = None
        x_122 = torch._C._nn.scaled_dot_product_attention(
            q_19, k_19, v_9, attn_mask=None, dropout_p=0.0
        )
        q_19 = k_19 = v_9 = None
        transpose_10 = x_122.transpose(1, 2)
        x_122 = None
        x_123 = transpose_10.reshape(1, 197, 1024)
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
        mul_58 = l_self_modules_blocks_modules_9_parameters_gamma_1_ * x_125
        l_self_modules_blocks_modules_9_parameters_gamma_1_ = x_125 = None
        x_126 = x_120 + mul_58
        x_120 = mul_58 = None
        x_127 = torch.nn.functional.layer_norm(
            x_126,
            (1024,),
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
        mul_59 = l_self_modules_blocks_modules_9_parameters_gamma_2_ * x_132
        l_self_modules_blocks_modules_9_parameters_gamma_2_ = x_132 = None
        x_133 = x_126 + mul_59
        x_126 = mul_59 = None
        x_134 = torch.nn.functional.layer_norm(
            x_133,
            (1024,),
            l_self_modules_blocks_modules_10_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_10_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_10_modules_norm1_parameters_bias_
        ) = None
        qkv_20 = torch._C._nn.linear(
            x_134,
            l_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_bias_,
        )
        x_134 = (
            l_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_40 = qkv_20.reshape(1, 197, 3, 16, -1)
        qkv_20 = None
        qkv_21 = reshape_40.permute(2, 0, 3, 1, 4)
        reshape_40 = None
        unbind_10 = qkv_21.unbind(0)
        qkv_21 = None
        q_20 = unbind_10[0]
        k_20 = unbind_10[1]
        v_10 = unbind_10[2]
        unbind_10 = None
        getitem_157 = q_20[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_158 = q_20[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_20 = None
        tensor_split_20 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_20 = tensor_split_20[0]
        cos_emb_20 = tensor_split_20[1]
        tensor_split_20 = None
        mul_60 = getitem_158 * cos_emb_20
        cos_emb_20 = None
        getitem_161 = getitem_158[(Ellipsis, slice(1, None, 2))]
        neg_20 = -getitem_161
        getitem_161 = None
        getitem_162 = getitem_158[(Ellipsis, slice(None, None, 2))]
        getitem_158 = None
        stack_20 = torch.stack([neg_20, getitem_162], -1)
        neg_20 = getitem_162 = None
        reshape_41 = stack_20.reshape((1, 16, 196, 64))
        stack_20 = None
        mul_61 = reshape_41 * sin_emb_20
        reshape_41 = sin_emb_20 = None
        add_40 = mul_60 + mul_61
        mul_60 = mul_61 = None
        cat_21 = torch.cat([getitem_157, add_40], dim=2)
        getitem_157 = add_40 = None
        q_21 = cat_21.type_as(v_10)
        cat_21 = None
        getitem_163 = k_20[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_164 = k_20[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_20 = None
        tensor_split_21 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_21 = tensor_split_21[0]
        cos_emb_21 = tensor_split_21[1]
        tensor_split_21 = None
        mul_62 = getitem_164 * cos_emb_21
        cos_emb_21 = None
        getitem_167 = getitem_164[(Ellipsis, slice(1, None, 2))]
        neg_21 = -getitem_167
        getitem_167 = None
        getitem_168 = getitem_164[(Ellipsis, slice(None, None, 2))]
        getitem_164 = None
        stack_21 = torch.stack([neg_21, getitem_168], -1)
        neg_21 = getitem_168 = None
        reshape_42 = stack_21.reshape((1, 16, 196, 64))
        stack_21 = None
        mul_63 = reshape_42 * sin_emb_21
        reshape_42 = sin_emb_21 = None
        add_41 = mul_62 + mul_63
        mul_62 = mul_63 = None
        cat_22 = torch.cat([getitem_163, add_41], dim=2)
        getitem_163 = add_41 = None
        k_21 = cat_22.type_as(v_10)
        cat_22 = None
        x_135 = torch._C._nn.scaled_dot_product_attention(
            q_21, k_21, v_10, attn_mask=None, dropout_p=0.0
        )
        q_21 = k_21 = v_10 = None
        transpose_11 = x_135.transpose(1, 2)
        x_135 = None
        x_136 = transpose_11.reshape(1, 197, 1024)
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
        mul_64 = l_self_modules_blocks_modules_10_parameters_gamma_1_ * x_138
        l_self_modules_blocks_modules_10_parameters_gamma_1_ = x_138 = None
        x_139 = x_133 + mul_64
        x_133 = mul_64 = None
        x_140 = torch.nn.functional.layer_norm(
            x_139,
            (1024,),
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
        mul_65 = l_self_modules_blocks_modules_10_parameters_gamma_2_ * x_145
        l_self_modules_blocks_modules_10_parameters_gamma_2_ = x_145 = None
        x_146 = x_139 + mul_65
        x_139 = mul_65 = None
        x_147 = torch.nn.functional.layer_norm(
            x_146,
            (1024,),
            l_self_modules_blocks_modules_11_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_11_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_11_modules_norm1_parameters_bias_
        ) = None
        qkv_22 = torch._C._nn.linear(
            x_147,
            l_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_bias_,
        )
        x_147 = (
            l_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_44 = qkv_22.reshape(1, 197, 3, 16, -1)
        qkv_22 = None
        qkv_23 = reshape_44.permute(2, 0, 3, 1, 4)
        reshape_44 = None
        unbind_11 = qkv_23.unbind(0)
        qkv_23 = None
        q_22 = unbind_11[0]
        k_22 = unbind_11[1]
        v_11 = unbind_11[2]
        unbind_11 = None
        getitem_172 = q_22[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_173 = q_22[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_22 = None
        tensor_split_22 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_22 = tensor_split_22[0]
        cos_emb_22 = tensor_split_22[1]
        tensor_split_22 = None
        mul_66 = getitem_173 * cos_emb_22
        cos_emb_22 = None
        getitem_176 = getitem_173[(Ellipsis, slice(1, None, 2))]
        neg_22 = -getitem_176
        getitem_176 = None
        getitem_177 = getitem_173[(Ellipsis, slice(None, None, 2))]
        getitem_173 = None
        stack_22 = torch.stack([neg_22, getitem_177], -1)
        neg_22 = getitem_177 = None
        reshape_45 = stack_22.reshape((1, 16, 196, 64))
        stack_22 = None
        mul_67 = reshape_45 * sin_emb_22
        reshape_45 = sin_emb_22 = None
        add_44 = mul_66 + mul_67
        mul_66 = mul_67 = None
        cat_23 = torch.cat([getitem_172, add_44], dim=2)
        getitem_172 = add_44 = None
        q_23 = cat_23.type_as(v_11)
        cat_23 = None
        getitem_178 = k_22[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_179 = k_22[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_22 = None
        tensor_split_23 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_23 = tensor_split_23[0]
        cos_emb_23 = tensor_split_23[1]
        tensor_split_23 = None
        mul_68 = getitem_179 * cos_emb_23
        cos_emb_23 = None
        getitem_182 = getitem_179[(Ellipsis, slice(1, None, 2))]
        neg_23 = -getitem_182
        getitem_182 = None
        getitem_183 = getitem_179[(Ellipsis, slice(None, None, 2))]
        getitem_179 = None
        stack_23 = torch.stack([neg_23, getitem_183], -1)
        neg_23 = getitem_183 = None
        reshape_46 = stack_23.reshape((1, 16, 196, 64))
        stack_23 = None
        mul_69 = reshape_46 * sin_emb_23
        reshape_46 = sin_emb_23 = None
        add_45 = mul_68 + mul_69
        mul_68 = mul_69 = None
        cat_24 = torch.cat([getitem_178, add_45], dim=2)
        getitem_178 = add_45 = None
        k_23 = cat_24.type_as(v_11)
        cat_24 = None
        x_148 = torch._C._nn.scaled_dot_product_attention(
            q_23, k_23, v_11, attn_mask=None, dropout_p=0.0
        )
        q_23 = k_23 = v_11 = None
        transpose_12 = x_148.transpose(1, 2)
        x_148 = None
        x_149 = transpose_12.reshape(1, 197, 1024)
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
        mul_70 = l_self_modules_blocks_modules_11_parameters_gamma_1_ * x_151
        l_self_modules_blocks_modules_11_parameters_gamma_1_ = x_151 = None
        x_152 = x_146 + mul_70
        x_146 = mul_70 = None
        x_153 = torch.nn.functional.layer_norm(
            x_152,
            (1024,),
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
        mul_71 = l_self_modules_blocks_modules_11_parameters_gamma_2_ * x_158
        l_self_modules_blocks_modules_11_parameters_gamma_2_ = x_158 = None
        x_159 = x_152 + mul_71
        x_152 = mul_71 = None
        x_160 = torch.nn.functional.layer_norm(
            x_159,
            (1024,),
            l_self_modules_blocks_modules_12_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_12_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_12_modules_norm1_parameters_bias_
        ) = None
        qkv_24 = torch._C._nn.linear(
            x_160,
            l_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_bias_,
        )
        x_160 = (
            l_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_48 = qkv_24.reshape(1, 197, 3, 16, -1)
        qkv_24 = None
        qkv_25 = reshape_48.permute(2, 0, 3, 1, 4)
        reshape_48 = None
        unbind_12 = qkv_25.unbind(0)
        qkv_25 = None
        q_24 = unbind_12[0]
        k_24 = unbind_12[1]
        v_12 = unbind_12[2]
        unbind_12 = None
        getitem_187 = q_24[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_188 = q_24[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_24 = None
        tensor_split_24 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_24 = tensor_split_24[0]
        cos_emb_24 = tensor_split_24[1]
        tensor_split_24 = None
        mul_72 = getitem_188 * cos_emb_24
        cos_emb_24 = None
        getitem_191 = getitem_188[(Ellipsis, slice(1, None, 2))]
        neg_24 = -getitem_191
        getitem_191 = None
        getitem_192 = getitem_188[(Ellipsis, slice(None, None, 2))]
        getitem_188 = None
        stack_24 = torch.stack([neg_24, getitem_192], -1)
        neg_24 = getitem_192 = None
        reshape_49 = stack_24.reshape((1, 16, 196, 64))
        stack_24 = None
        mul_73 = reshape_49 * sin_emb_24
        reshape_49 = sin_emb_24 = None
        add_48 = mul_72 + mul_73
        mul_72 = mul_73 = None
        cat_25 = torch.cat([getitem_187, add_48], dim=2)
        getitem_187 = add_48 = None
        q_25 = cat_25.type_as(v_12)
        cat_25 = None
        getitem_193 = k_24[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_194 = k_24[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_24 = None
        tensor_split_25 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_25 = tensor_split_25[0]
        cos_emb_25 = tensor_split_25[1]
        tensor_split_25 = None
        mul_74 = getitem_194 * cos_emb_25
        cos_emb_25 = None
        getitem_197 = getitem_194[(Ellipsis, slice(1, None, 2))]
        neg_25 = -getitem_197
        getitem_197 = None
        getitem_198 = getitem_194[(Ellipsis, slice(None, None, 2))]
        getitem_194 = None
        stack_25 = torch.stack([neg_25, getitem_198], -1)
        neg_25 = getitem_198 = None
        reshape_50 = stack_25.reshape((1, 16, 196, 64))
        stack_25 = None
        mul_75 = reshape_50 * sin_emb_25
        reshape_50 = sin_emb_25 = None
        add_49 = mul_74 + mul_75
        mul_74 = mul_75 = None
        cat_26 = torch.cat([getitem_193, add_49], dim=2)
        getitem_193 = add_49 = None
        k_25 = cat_26.type_as(v_12)
        cat_26 = None
        x_161 = torch._C._nn.scaled_dot_product_attention(
            q_25, k_25, v_12, attn_mask=None, dropout_p=0.0
        )
        q_25 = k_25 = v_12 = None
        transpose_13 = x_161.transpose(1, 2)
        x_161 = None
        x_162 = transpose_13.reshape(1, 197, 1024)
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
        mul_76 = l_self_modules_blocks_modules_12_parameters_gamma_1_ * x_164
        l_self_modules_blocks_modules_12_parameters_gamma_1_ = x_164 = None
        x_165 = x_159 + mul_76
        x_159 = mul_76 = None
        x_166 = torch.nn.functional.layer_norm(
            x_165,
            (1024,),
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
        mul_77 = l_self_modules_blocks_modules_12_parameters_gamma_2_ * x_171
        l_self_modules_blocks_modules_12_parameters_gamma_2_ = x_171 = None
        x_172 = x_165 + mul_77
        x_165 = mul_77 = None
        x_173 = torch.nn.functional.layer_norm(
            x_172,
            (1024,),
            l_self_modules_blocks_modules_13_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_13_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_13_modules_norm1_parameters_bias_
        ) = None
        qkv_26 = torch._C._nn.linear(
            x_173,
            l_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_bias_,
        )
        x_173 = (
            l_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_52 = qkv_26.reshape(1, 197, 3, 16, -1)
        qkv_26 = None
        qkv_27 = reshape_52.permute(2, 0, 3, 1, 4)
        reshape_52 = None
        unbind_13 = qkv_27.unbind(0)
        qkv_27 = None
        q_26 = unbind_13[0]
        k_26 = unbind_13[1]
        v_13 = unbind_13[2]
        unbind_13 = None
        getitem_202 = q_26[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_203 = q_26[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_26 = None
        tensor_split_26 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_26 = tensor_split_26[0]
        cos_emb_26 = tensor_split_26[1]
        tensor_split_26 = None
        mul_78 = getitem_203 * cos_emb_26
        cos_emb_26 = None
        getitem_206 = getitem_203[(Ellipsis, slice(1, None, 2))]
        neg_26 = -getitem_206
        getitem_206 = None
        getitem_207 = getitem_203[(Ellipsis, slice(None, None, 2))]
        getitem_203 = None
        stack_26 = torch.stack([neg_26, getitem_207], -1)
        neg_26 = getitem_207 = None
        reshape_53 = stack_26.reshape((1, 16, 196, 64))
        stack_26 = None
        mul_79 = reshape_53 * sin_emb_26
        reshape_53 = sin_emb_26 = None
        add_52 = mul_78 + mul_79
        mul_78 = mul_79 = None
        cat_27 = torch.cat([getitem_202, add_52], dim=2)
        getitem_202 = add_52 = None
        q_27 = cat_27.type_as(v_13)
        cat_27 = None
        getitem_208 = k_26[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_209 = k_26[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_26 = None
        tensor_split_27 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_27 = tensor_split_27[0]
        cos_emb_27 = tensor_split_27[1]
        tensor_split_27 = None
        mul_80 = getitem_209 * cos_emb_27
        cos_emb_27 = None
        getitem_212 = getitem_209[(Ellipsis, slice(1, None, 2))]
        neg_27 = -getitem_212
        getitem_212 = None
        getitem_213 = getitem_209[(Ellipsis, slice(None, None, 2))]
        getitem_209 = None
        stack_27 = torch.stack([neg_27, getitem_213], -1)
        neg_27 = getitem_213 = None
        reshape_54 = stack_27.reshape((1, 16, 196, 64))
        stack_27 = None
        mul_81 = reshape_54 * sin_emb_27
        reshape_54 = sin_emb_27 = None
        add_53 = mul_80 + mul_81
        mul_80 = mul_81 = None
        cat_28 = torch.cat([getitem_208, add_53], dim=2)
        getitem_208 = add_53 = None
        k_27 = cat_28.type_as(v_13)
        cat_28 = None
        x_174 = torch._C._nn.scaled_dot_product_attention(
            q_27, k_27, v_13, attn_mask=None, dropout_p=0.0
        )
        q_27 = k_27 = v_13 = None
        transpose_14 = x_174.transpose(1, 2)
        x_174 = None
        x_175 = transpose_14.reshape(1, 197, 1024)
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
        mul_82 = l_self_modules_blocks_modules_13_parameters_gamma_1_ * x_177
        l_self_modules_blocks_modules_13_parameters_gamma_1_ = x_177 = None
        x_178 = x_172 + mul_82
        x_172 = mul_82 = None
        x_179 = torch.nn.functional.layer_norm(
            x_178,
            (1024,),
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
        mul_83 = l_self_modules_blocks_modules_13_parameters_gamma_2_ * x_184
        l_self_modules_blocks_modules_13_parameters_gamma_2_ = x_184 = None
        x_185 = x_178 + mul_83
        x_178 = mul_83 = None
        x_186 = torch.nn.functional.layer_norm(
            x_185,
            (1024,),
            l_self_modules_blocks_modules_14_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_14_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_14_modules_norm1_parameters_bias_
        ) = None
        qkv_28 = torch._C._nn.linear(
            x_186,
            l_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_bias_,
        )
        x_186 = (
            l_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_56 = qkv_28.reshape(1, 197, 3, 16, -1)
        qkv_28 = None
        qkv_29 = reshape_56.permute(2, 0, 3, 1, 4)
        reshape_56 = None
        unbind_14 = qkv_29.unbind(0)
        qkv_29 = None
        q_28 = unbind_14[0]
        k_28 = unbind_14[1]
        v_14 = unbind_14[2]
        unbind_14 = None
        getitem_217 = q_28[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_218 = q_28[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_28 = None
        tensor_split_28 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_28 = tensor_split_28[0]
        cos_emb_28 = tensor_split_28[1]
        tensor_split_28 = None
        mul_84 = getitem_218 * cos_emb_28
        cos_emb_28 = None
        getitem_221 = getitem_218[(Ellipsis, slice(1, None, 2))]
        neg_28 = -getitem_221
        getitem_221 = None
        getitem_222 = getitem_218[(Ellipsis, slice(None, None, 2))]
        getitem_218 = None
        stack_28 = torch.stack([neg_28, getitem_222], -1)
        neg_28 = getitem_222 = None
        reshape_57 = stack_28.reshape((1, 16, 196, 64))
        stack_28 = None
        mul_85 = reshape_57 * sin_emb_28
        reshape_57 = sin_emb_28 = None
        add_56 = mul_84 + mul_85
        mul_84 = mul_85 = None
        cat_29 = torch.cat([getitem_217, add_56], dim=2)
        getitem_217 = add_56 = None
        q_29 = cat_29.type_as(v_14)
        cat_29 = None
        getitem_223 = k_28[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_224 = k_28[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_28 = None
        tensor_split_29 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_29 = tensor_split_29[0]
        cos_emb_29 = tensor_split_29[1]
        tensor_split_29 = None
        mul_86 = getitem_224 * cos_emb_29
        cos_emb_29 = None
        getitem_227 = getitem_224[(Ellipsis, slice(1, None, 2))]
        neg_29 = -getitem_227
        getitem_227 = None
        getitem_228 = getitem_224[(Ellipsis, slice(None, None, 2))]
        getitem_224 = None
        stack_29 = torch.stack([neg_29, getitem_228], -1)
        neg_29 = getitem_228 = None
        reshape_58 = stack_29.reshape((1, 16, 196, 64))
        stack_29 = None
        mul_87 = reshape_58 * sin_emb_29
        reshape_58 = sin_emb_29 = None
        add_57 = mul_86 + mul_87
        mul_86 = mul_87 = None
        cat_30 = torch.cat([getitem_223, add_57], dim=2)
        getitem_223 = add_57 = None
        k_29 = cat_30.type_as(v_14)
        cat_30 = None
        x_187 = torch._C._nn.scaled_dot_product_attention(
            q_29, k_29, v_14, attn_mask=None, dropout_p=0.0
        )
        q_29 = k_29 = v_14 = None
        transpose_15 = x_187.transpose(1, 2)
        x_187 = None
        x_188 = transpose_15.reshape(1, 197, 1024)
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
        mul_88 = l_self_modules_blocks_modules_14_parameters_gamma_1_ * x_190
        l_self_modules_blocks_modules_14_parameters_gamma_1_ = x_190 = None
        x_191 = x_185 + mul_88
        x_185 = mul_88 = None
        x_192 = torch.nn.functional.layer_norm(
            x_191,
            (1024,),
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
        mul_89 = l_self_modules_blocks_modules_14_parameters_gamma_2_ * x_197
        l_self_modules_blocks_modules_14_parameters_gamma_2_ = x_197 = None
        x_198 = x_191 + mul_89
        x_191 = mul_89 = None
        x_199 = torch.nn.functional.layer_norm(
            x_198,
            (1024,),
            l_self_modules_blocks_modules_15_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_15_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_15_modules_norm1_parameters_bias_
        ) = None
        qkv_30 = torch._C._nn.linear(
            x_199,
            l_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_bias_,
        )
        x_199 = (
            l_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_60 = qkv_30.reshape(1, 197, 3, 16, -1)
        qkv_30 = None
        qkv_31 = reshape_60.permute(2, 0, 3, 1, 4)
        reshape_60 = None
        unbind_15 = qkv_31.unbind(0)
        qkv_31 = None
        q_30 = unbind_15[0]
        k_30 = unbind_15[1]
        v_15 = unbind_15[2]
        unbind_15 = None
        getitem_232 = q_30[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_233 = q_30[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_30 = None
        tensor_split_30 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_30 = tensor_split_30[0]
        cos_emb_30 = tensor_split_30[1]
        tensor_split_30 = None
        mul_90 = getitem_233 * cos_emb_30
        cos_emb_30 = None
        getitem_236 = getitem_233[(Ellipsis, slice(1, None, 2))]
        neg_30 = -getitem_236
        getitem_236 = None
        getitem_237 = getitem_233[(Ellipsis, slice(None, None, 2))]
        getitem_233 = None
        stack_30 = torch.stack([neg_30, getitem_237], -1)
        neg_30 = getitem_237 = None
        reshape_61 = stack_30.reshape((1, 16, 196, 64))
        stack_30 = None
        mul_91 = reshape_61 * sin_emb_30
        reshape_61 = sin_emb_30 = None
        add_60 = mul_90 + mul_91
        mul_90 = mul_91 = None
        cat_31 = torch.cat([getitem_232, add_60], dim=2)
        getitem_232 = add_60 = None
        q_31 = cat_31.type_as(v_15)
        cat_31 = None
        getitem_238 = k_30[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_239 = k_30[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_30 = None
        tensor_split_31 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_31 = tensor_split_31[0]
        cos_emb_31 = tensor_split_31[1]
        tensor_split_31 = None
        mul_92 = getitem_239 * cos_emb_31
        cos_emb_31 = None
        getitem_242 = getitem_239[(Ellipsis, slice(1, None, 2))]
        neg_31 = -getitem_242
        getitem_242 = None
        getitem_243 = getitem_239[(Ellipsis, slice(None, None, 2))]
        getitem_239 = None
        stack_31 = torch.stack([neg_31, getitem_243], -1)
        neg_31 = getitem_243 = None
        reshape_62 = stack_31.reshape((1, 16, 196, 64))
        stack_31 = None
        mul_93 = reshape_62 * sin_emb_31
        reshape_62 = sin_emb_31 = None
        add_61 = mul_92 + mul_93
        mul_92 = mul_93 = None
        cat_32 = torch.cat([getitem_238, add_61], dim=2)
        getitem_238 = add_61 = None
        k_31 = cat_32.type_as(v_15)
        cat_32 = None
        x_200 = torch._C._nn.scaled_dot_product_attention(
            q_31, k_31, v_15, attn_mask=None, dropout_p=0.0
        )
        q_31 = k_31 = v_15 = None
        transpose_16 = x_200.transpose(1, 2)
        x_200 = None
        x_201 = transpose_16.reshape(1, 197, 1024)
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
        mul_94 = l_self_modules_blocks_modules_15_parameters_gamma_1_ * x_203
        l_self_modules_blocks_modules_15_parameters_gamma_1_ = x_203 = None
        x_204 = x_198 + mul_94
        x_198 = mul_94 = None
        x_205 = torch.nn.functional.layer_norm(
            x_204,
            (1024,),
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
        mul_95 = l_self_modules_blocks_modules_15_parameters_gamma_2_ * x_210
        l_self_modules_blocks_modules_15_parameters_gamma_2_ = x_210 = None
        x_211 = x_204 + mul_95
        x_204 = mul_95 = None
        x_212 = torch.nn.functional.layer_norm(
            x_211,
            (1024,),
            l_self_modules_blocks_modules_16_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_16_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_16_modules_norm1_parameters_bias_
        ) = None
        qkv_32 = torch._C._nn.linear(
            x_212,
            l_self_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_bias_,
        )
        x_212 = (
            l_self_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_64 = qkv_32.reshape(1, 197, 3, 16, -1)
        qkv_32 = None
        qkv_33 = reshape_64.permute(2, 0, 3, 1, 4)
        reshape_64 = None
        unbind_16 = qkv_33.unbind(0)
        qkv_33 = None
        q_32 = unbind_16[0]
        k_32 = unbind_16[1]
        v_16 = unbind_16[2]
        unbind_16 = None
        getitem_247 = q_32[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_248 = q_32[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_32 = None
        tensor_split_32 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_32 = tensor_split_32[0]
        cos_emb_32 = tensor_split_32[1]
        tensor_split_32 = None
        mul_96 = getitem_248 * cos_emb_32
        cos_emb_32 = None
        getitem_251 = getitem_248[(Ellipsis, slice(1, None, 2))]
        neg_32 = -getitem_251
        getitem_251 = None
        getitem_252 = getitem_248[(Ellipsis, slice(None, None, 2))]
        getitem_248 = None
        stack_32 = torch.stack([neg_32, getitem_252], -1)
        neg_32 = getitem_252 = None
        reshape_65 = stack_32.reshape((1, 16, 196, 64))
        stack_32 = None
        mul_97 = reshape_65 * sin_emb_32
        reshape_65 = sin_emb_32 = None
        add_64 = mul_96 + mul_97
        mul_96 = mul_97 = None
        cat_33 = torch.cat([getitem_247, add_64], dim=2)
        getitem_247 = add_64 = None
        q_33 = cat_33.type_as(v_16)
        cat_33 = None
        getitem_253 = k_32[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_254 = k_32[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_32 = None
        tensor_split_33 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_33 = tensor_split_33[0]
        cos_emb_33 = tensor_split_33[1]
        tensor_split_33 = None
        mul_98 = getitem_254 * cos_emb_33
        cos_emb_33 = None
        getitem_257 = getitem_254[(Ellipsis, slice(1, None, 2))]
        neg_33 = -getitem_257
        getitem_257 = None
        getitem_258 = getitem_254[(Ellipsis, slice(None, None, 2))]
        getitem_254 = None
        stack_33 = torch.stack([neg_33, getitem_258], -1)
        neg_33 = getitem_258 = None
        reshape_66 = stack_33.reshape((1, 16, 196, 64))
        stack_33 = None
        mul_99 = reshape_66 * sin_emb_33
        reshape_66 = sin_emb_33 = None
        add_65 = mul_98 + mul_99
        mul_98 = mul_99 = None
        cat_34 = torch.cat([getitem_253, add_65], dim=2)
        getitem_253 = add_65 = None
        k_33 = cat_34.type_as(v_16)
        cat_34 = None
        x_213 = torch._C._nn.scaled_dot_product_attention(
            q_33, k_33, v_16, attn_mask=None, dropout_p=0.0
        )
        q_33 = k_33 = v_16 = None
        transpose_17 = x_213.transpose(1, 2)
        x_213 = None
        x_214 = transpose_17.reshape(1, 197, 1024)
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
        mul_100 = l_self_modules_blocks_modules_16_parameters_gamma_1_ * x_216
        l_self_modules_blocks_modules_16_parameters_gamma_1_ = x_216 = None
        x_217 = x_211 + mul_100
        x_211 = mul_100 = None
        x_218 = torch.nn.functional.layer_norm(
            x_217,
            (1024,),
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
        mul_101 = l_self_modules_blocks_modules_16_parameters_gamma_2_ * x_223
        l_self_modules_blocks_modules_16_parameters_gamma_2_ = x_223 = None
        x_224 = x_217 + mul_101
        x_217 = mul_101 = None
        x_225 = torch.nn.functional.layer_norm(
            x_224,
            (1024,),
            l_self_modules_blocks_modules_17_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_17_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_17_modules_norm1_parameters_bias_
        ) = None
        qkv_34 = torch._C._nn.linear(
            x_225,
            l_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_bias_,
        )
        x_225 = (
            l_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_68 = qkv_34.reshape(1, 197, 3, 16, -1)
        qkv_34 = None
        qkv_35 = reshape_68.permute(2, 0, 3, 1, 4)
        reshape_68 = None
        unbind_17 = qkv_35.unbind(0)
        qkv_35 = None
        q_34 = unbind_17[0]
        k_34 = unbind_17[1]
        v_17 = unbind_17[2]
        unbind_17 = None
        getitem_262 = q_34[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_263 = q_34[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_34 = None
        tensor_split_34 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_34 = tensor_split_34[0]
        cos_emb_34 = tensor_split_34[1]
        tensor_split_34 = None
        mul_102 = getitem_263 * cos_emb_34
        cos_emb_34 = None
        getitem_266 = getitem_263[(Ellipsis, slice(1, None, 2))]
        neg_34 = -getitem_266
        getitem_266 = None
        getitem_267 = getitem_263[(Ellipsis, slice(None, None, 2))]
        getitem_263 = None
        stack_34 = torch.stack([neg_34, getitem_267], -1)
        neg_34 = getitem_267 = None
        reshape_69 = stack_34.reshape((1, 16, 196, 64))
        stack_34 = None
        mul_103 = reshape_69 * sin_emb_34
        reshape_69 = sin_emb_34 = None
        add_68 = mul_102 + mul_103
        mul_102 = mul_103 = None
        cat_35 = torch.cat([getitem_262, add_68], dim=2)
        getitem_262 = add_68 = None
        q_35 = cat_35.type_as(v_17)
        cat_35 = None
        getitem_268 = k_34[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_269 = k_34[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_34 = None
        tensor_split_35 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_35 = tensor_split_35[0]
        cos_emb_35 = tensor_split_35[1]
        tensor_split_35 = None
        mul_104 = getitem_269 * cos_emb_35
        cos_emb_35 = None
        getitem_272 = getitem_269[(Ellipsis, slice(1, None, 2))]
        neg_35 = -getitem_272
        getitem_272 = None
        getitem_273 = getitem_269[(Ellipsis, slice(None, None, 2))]
        getitem_269 = None
        stack_35 = torch.stack([neg_35, getitem_273], -1)
        neg_35 = getitem_273 = None
        reshape_70 = stack_35.reshape((1, 16, 196, 64))
        stack_35 = None
        mul_105 = reshape_70 * sin_emb_35
        reshape_70 = sin_emb_35 = None
        add_69 = mul_104 + mul_105
        mul_104 = mul_105 = None
        cat_36 = torch.cat([getitem_268, add_69], dim=2)
        getitem_268 = add_69 = None
        k_35 = cat_36.type_as(v_17)
        cat_36 = None
        x_226 = torch._C._nn.scaled_dot_product_attention(
            q_35, k_35, v_17, attn_mask=None, dropout_p=0.0
        )
        q_35 = k_35 = v_17 = None
        transpose_18 = x_226.transpose(1, 2)
        x_226 = None
        x_227 = transpose_18.reshape(1, 197, 1024)
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
        mul_106 = l_self_modules_blocks_modules_17_parameters_gamma_1_ * x_229
        l_self_modules_blocks_modules_17_parameters_gamma_1_ = x_229 = None
        x_230 = x_224 + mul_106
        x_224 = mul_106 = None
        x_231 = torch.nn.functional.layer_norm(
            x_230,
            (1024,),
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
        mul_107 = l_self_modules_blocks_modules_17_parameters_gamma_2_ * x_236
        l_self_modules_blocks_modules_17_parameters_gamma_2_ = x_236 = None
        x_237 = x_230 + mul_107
        x_230 = mul_107 = None
        x_238 = torch.nn.functional.layer_norm(
            x_237,
            (1024,),
            l_self_modules_blocks_modules_18_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_18_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_18_modules_norm1_parameters_bias_
        ) = None
        qkv_36 = torch._C._nn.linear(
            x_238,
            l_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_bias_,
        )
        x_238 = (
            l_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_72 = qkv_36.reshape(1, 197, 3, 16, -1)
        qkv_36 = None
        qkv_37 = reshape_72.permute(2, 0, 3, 1, 4)
        reshape_72 = None
        unbind_18 = qkv_37.unbind(0)
        qkv_37 = None
        q_36 = unbind_18[0]
        k_36 = unbind_18[1]
        v_18 = unbind_18[2]
        unbind_18 = None
        getitem_277 = q_36[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_278 = q_36[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_36 = None
        tensor_split_36 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_36 = tensor_split_36[0]
        cos_emb_36 = tensor_split_36[1]
        tensor_split_36 = None
        mul_108 = getitem_278 * cos_emb_36
        cos_emb_36 = None
        getitem_281 = getitem_278[(Ellipsis, slice(1, None, 2))]
        neg_36 = -getitem_281
        getitem_281 = None
        getitem_282 = getitem_278[(Ellipsis, slice(None, None, 2))]
        getitem_278 = None
        stack_36 = torch.stack([neg_36, getitem_282], -1)
        neg_36 = getitem_282 = None
        reshape_73 = stack_36.reshape((1, 16, 196, 64))
        stack_36 = None
        mul_109 = reshape_73 * sin_emb_36
        reshape_73 = sin_emb_36 = None
        add_72 = mul_108 + mul_109
        mul_108 = mul_109 = None
        cat_37 = torch.cat([getitem_277, add_72], dim=2)
        getitem_277 = add_72 = None
        q_37 = cat_37.type_as(v_18)
        cat_37 = None
        getitem_283 = k_36[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_284 = k_36[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_36 = None
        tensor_split_37 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_37 = tensor_split_37[0]
        cos_emb_37 = tensor_split_37[1]
        tensor_split_37 = None
        mul_110 = getitem_284 * cos_emb_37
        cos_emb_37 = None
        getitem_287 = getitem_284[(Ellipsis, slice(1, None, 2))]
        neg_37 = -getitem_287
        getitem_287 = None
        getitem_288 = getitem_284[(Ellipsis, slice(None, None, 2))]
        getitem_284 = None
        stack_37 = torch.stack([neg_37, getitem_288], -1)
        neg_37 = getitem_288 = None
        reshape_74 = stack_37.reshape((1, 16, 196, 64))
        stack_37 = None
        mul_111 = reshape_74 * sin_emb_37
        reshape_74 = sin_emb_37 = None
        add_73 = mul_110 + mul_111
        mul_110 = mul_111 = None
        cat_38 = torch.cat([getitem_283, add_73], dim=2)
        getitem_283 = add_73 = None
        k_37 = cat_38.type_as(v_18)
        cat_38 = None
        x_239 = torch._C._nn.scaled_dot_product_attention(
            q_37, k_37, v_18, attn_mask=None, dropout_p=0.0
        )
        q_37 = k_37 = v_18 = None
        transpose_19 = x_239.transpose(1, 2)
        x_239 = None
        x_240 = transpose_19.reshape(1, 197, 1024)
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
        mul_112 = l_self_modules_blocks_modules_18_parameters_gamma_1_ * x_242
        l_self_modules_blocks_modules_18_parameters_gamma_1_ = x_242 = None
        x_243 = x_237 + mul_112
        x_237 = mul_112 = None
        x_244 = torch.nn.functional.layer_norm(
            x_243,
            (1024,),
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
        mul_113 = l_self_modules_blocks_modules_18_parameters_gamma_2_ * x_249
        l_self_modules_blocks_modules_18_parameters_gamma_2_ = x_249 = None
        x_250 = x_243 + mul_113
        x_243 = mul_113 = None
        x_251 = torch.nn.functional.layer_norm(
            x_250,
            (1024,),
            l_self_modules_blocks_modules_19_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_19_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_19_modules_norm1_parameters_bias_
        ) = None
        qkv_38 = torch._C._nn.linear(
            x_251,
            l_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_bias_,
        )
        x_251 = (
            l_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_76 = qkv_38.reshape(1, 197, 3, 16, -1)
        qkv_38 = None
        qkv_39 = reshape_76.permute(2, 0, 3, 1, 4)
        reshape_76 = None
        unbind_19 = qkv_39.unbind(0)
        qkv_39 = None
        q_38 = unbind_19[0]
        k_38 = unbind_19[1]
        v_19 = unbind_19[2]
        unbind_19 = None
        getitem_292 = q_38[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_293 = q_38[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_38 = None
        tensor_split_38 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_38 = tensor_split_38[0]
        cos_emb_38 = tensor_split_38[1]
        tensor_split_38 = None
        mul_114 = getitem_293 * cos_emb_38
        cos_emb_38 = None
        getitem_296 = getitem_293[(Ellipsis, slice(1, None, 2))]
        neg_38 = -getitem_296
        getitem_296 = None
        getitem_297 = getitem_293[(Ellipsis, slice(None, None, 2))]
        getitem_293 = None
        stack_38 = torch.stack([neg_38, getitem_297], -1)
        neg_38 = getitem_297 = None
        reshape_77 = stack_38.reshape((1, 16, 196, 64))
        stack_38 = None
        mul_115 = reshape_77 * sin_emb_38
        reshape_77 = sin_emb_38 = None
        add_76 = mul_114 + mul_115
        mul_114 = mul_115 = None
        cat_39 = torch.cat([getitem_292, add_76], dim=2)
        getitem_292 = add_76 = None
        q_39 = cat_39.type_as(v_19)
        cat_39 = None
        getitem_298 = k_38[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_299 = k_38[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_38 = None
        tensor_split_39 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_39 = tensor_split_39[0]
        cos_emb_39 = tensor_split_39[1]
        tensor_split_39 = None
        mul_116 = getitem_299 * cos_emb_39
        cos_emb_39 = None
        getitem_302 = getitem_299[(Ellipsis, slice(1, None, 2))]
        neg_39 = -getitem_302
        getitem_302 = None
        getitem_303 = getitem_299[(Ellipsis, slice(None, None, 2))]
        getitem_299 = None
        stack_39 = torch.stack([neg_39, getitem_303], -1)
        neg_39 = getitem_303 = None
        reshape_78 = stack_39.reshape((1, 16, 196, 64))
        stack_39 = None
        mul_117 = reshape_78 * sin_emb_39
        reshape_78 = sin_emb_39 = None
        add_77 = mul_116 + mul_117
        mul_116 = mul_117 = None
        cat_40 = torch.cat([getitem_298, add_77], dim=2)
        getitem_298 = add_77 = None
        k_39 = cat_40.type_as(v_19)
        cat_40 = None
        x_252 = torch._C._nn.scaled_dot_product_attention(
            q_39, k_39, v_19, attn_mask=None, dropout_p=0.0
        )
        q_39 = k_39 = v_19 = None
        transpose_20 = x_252.transpose(1, 2)
        x_252 = None
        x_253 = transpose_20.reshape(1, 197, 1024)
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
        mul_118 = l_self_modules_blocks_modules_19_parameters_gamma_1_ * x_255
        l_self_modules_blocks_modules_19_parameters_gamma_1_ = x_255 = None
        x_256 = x_250 + mul_118
        x_250 = mul_118 = None
        x_257 = torch.nn.functional.layer_norm(
            x_256,
            (1024,),
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
        mul_119 = l_self_modules_blocks_modules_19_parameters_gamma_2_ * x_262
        l_self_modules_blocks_modules_19_parameters_gamma_2_ = x_262 = None
        x_263 = x_256 + mul_119
        x_256 = mul_119 = None
        x_264 = torch.nn.functional.layer_norm(
            x_263,
            (1024,),
            l_self_modules_blocks_modules_20_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_20_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_20_modules_norm1_parameters_bias_
        ) = None
        qkv_40 = torch._C._nn.linear(
            x_264,
            l_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_bias_,
        )
        x_264 = (
            l_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_80 = qkv_40.reshape(1, 197, 3, 16, -1)
        qkv_40 = None
        qkv_41 = reshape_80.permute(2, 0, 3, 1, 4)
        reshape_80 = None
        unbind_20 = qkv_41.unbind(0)
        qkv_41 = None
        q_40 = unbind_20[0]
        k_40 = unbind_20[1]
        v_20 = unbind_20[2]
        unbind_20 = None
        getitem_307 = q_40[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_308 = q_40[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_40 = None
        tensor_split_40 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_40 = tensor_split_40[0]
        cos_emb_40 = tensor_split_40[1]
        tensor_split_40 = None
        mul_120 = getitem_308 * cos_emb_40
        cos_emb_40 = None
        getitem_311 = getitem_308[(Ellipsis, slice(1, None, 2))]
        neg_40 = -getitem_311
        getitem_311 = None
        getitem_312 = getitem_308[(Ellipsis, slice(None, None, 2))]
        getitem_308 = None
        stack_40 = torch.stack([neg_40, getitem_312], -1)
        neg_40 = getitem_312 = None
        reshape_81 = stack_40.reshape((1, 16, 196, 64))
        stack_40 = None
        mul_121 = reshape_81 * sin_emb_40
        reshape_81 = sin_emb_40 = None
        add_80 = mul_120 + mul_121
        mul_120 = mul_121 = None
        cat_41 = torch.cat([getitem_307, add_80], dim=2)
        getitem_307 = add_80 = None
        q_41 = cat_41.type_as(v_20)
        cat_41 = None
        getitem_313 = k_40[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_314 = k_40[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_40 = None
        tensor_split_41 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_41 = tensor_split_41[0]
        cos_emb_41 = tensor_split_41[1]
        tensor_split_41 = None
        mul_122 = getitem_314 * cos_emb_41
        cos_emb_41 = None
        getitem_317 = getitem_314[(Ellipsis, slice(1, None, 2))]
        neg_41 = -getitem_317
        getitem_317 = None
        getitem_318 = getitem_314[(Ellipsis, slice(None, None, 2))]
        getitem_314 = None
        stack_41 = torch.stack([neg_41, getitem_318], -1)
        neg_41 = getitem_318 = None
        reshape_82 = stack_41.reshape((1, 16, 196, 64))
        stack_41 = None
        mul_123 = reshape_82 * sin_emb_41
        reshape_82 = sin_emb_41 = None
        add_81 = mul_122 + mul_123
        mul_122 = mul_123 = None
        cat_42 = torch.cat([getitem_313, add_81], dim=2)
        getitem_313 = add_81 = None
        k_41 = cat_42.type_as(v_20)
        cat_42 = None
        x_265 = torch._C._nn.scaled_dot_product_attention(
            q_41, k_41, v_20, attn_mask=None, dropout_p=0.0
        )
        q_41 = k_41 = v_20 = None
        transpose_21 = x_265.transpose(1, 2)
        x_265 = None
        x_266 = transpose_21.reshape(1, 197, 1024)
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
        mul_124 = l_self_modules_blocks_modules_20_parameters_gamma_1_ * x_268
        l_self_modules_blocks_modules_20_parameters_gamma_1_ = x_268 = None
        x_269 = x_263 + mul_124
        x_263 = mul_124 = None
        x_270 = torch.nn.functional.layer_norm(
            x_269,
            (1024,),
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
        mul_125 = l_self_modules_blocks_modules_20_parameters_gamma_2_ * x_275
        l_self_modules_blocks_modules_20_parameters_gamma_2_ = x_275 = None
        x_276 = x_269 + mul_125
        x_269 = mul_125 = None
        x_277 = torch.nn.functional.layer_norm(
            x_276,
            (1024,),
            l_self_modules_blocks_modules_21_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_21_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_21_modules_norm1_parameters_bias_
        ) = None
        qkv_42 = torch._C._nn.linear(
            x_277,
            l_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_bias_,
        )
        x_277 = (
            l_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_84 = qkv_42.reshape(1, 197, 3, 16, -1)
        qkv_42 = None
        qkv_43 = reshape_84.permute(2, 0, 3, 1, 4)
        reshape_84 = None
        unbind_21 = qkv_43.unbind(0)
        qkv_43 = None
        q_42 = unbind_21[0]
        k_42 = unbind_21[1]
        v_21 = unbind_21[2]
        unbind_21 = None
        getitem_322 = q_42[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_323 = q_42[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_42 = None
        tensor_split_42 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_42 = tensor_split_42[0]
        cos_emb_42 = tensor_split_42[1]
        tensor_split_42 = None
        mul_126 = getitem_323 * cos_emb_42
        cos_emb_42 = None
        getitem_326 = getitem_323[(Ellipsis, slice(1, None, 2))]
        neg_42 = -getitem_326
        getitem_326 = None
        getitem_327 = getitem_323[(Ellipsis, slice(None, None, 2))]
        getitem_323 = None
        stack_42 = torch.stack([neg_42, getitem_327], -1)
        neg_42 = getitem_327 = None
        reshape_85 = stack_42.reshape((1, 16, 196, 64))
        stack_42 = None
        mul_127 = reshape_85 * sin_emb_42
        reshape_85 = sin_emb_42 = None
        add_84 = mul_126 + mul_127
        mul_126 = mul_127 = None
        cat_43 = torch.cat([getitem_322, add_84], dim=2)
        getitem_322 = add_84 = None
        q_43 = cat_43.type_as(v_21)
        cat_43 = None
        getitem_328 = k_42[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_329 = k_42[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_42 = None
        tensor_split_43 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_43 = tensor_split_43[0]
        cos_emb_43 = tensor_split_43[1]
        tensor_split_43 = None
        mul_128 = getitem_329 * cos_emb_43
        cos_emb_43 = None
        getitem_332 = getitem_329[(Ellipsis, slice(1, None, 2))]
        neg_43 = -getitem_332
        getitem_332 = None
        getitem_333 = getitem_329[(Ellipsis, slice(None, None, 2))]
        getitem_329 = None
        stack_43 = torch.stack([neg_43, getitem_333], -1)
        neg_43 = getitem_333 = None
        reshape_86 = stack_43.reshape((1, 16, 196, 64))
        stack_43 = None
        mul_129 = reshape_86 * sin_emb_43
        reshape_86 = sin_emb_43 = None
        add_85 = mul_128 + mul_129
        mul_128 = mul_129 = None
        cat_44 = torch.cat([getitem_328, add_85], dim=2)
        getitem_328 = add_85 = None
        k_43 = cat_44.type_as(v_21)
        cat_44 = None
        x_278 = torch._C._nn.scaled_dot_product_attention(
            q_43, k_43, v_21, attn_mask=None, dropout_p=0.0
        )
        q_43 = k_43 = v_21 = None
        transpose_22 = x_278.transpose(1, 2)
        x_278 = None
        x_279 = transpose_22.reshape(1, 197, 1024)
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
        mul_130 = l_self_modules_blocks_modules_21_parameters_gamma_1_ * x_281
        l_self_modules_blocks_modules_21_parameters_gamma_1_ = x_281 = None
        x_282 = x_276 + mul_130
        x_276 = mul_130 = None
        x_283 = torch.nn.functional.layer_norm(
            x_282,
            (1024,),
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
        mul_131 = l_self_modules_blocks_modules_21_parameters_gamma_2_ * x_288
        l_self_modules_blocks_modules_21_parameters_gamma_2_ = x_288 = None
        x_289 = x_282 + mul_131
        x_282 = mul_131 = None
        x_290 = torch.nn.functional.layer_norm(
            x_289,
            (1024,),
            l_self_modules_blocks_modules_22_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_22_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_22_modules_norm1_parameters_bias_
        ) = None
        qkv_44 = torch._C._nn.linear(
            x_290,
            l_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_bias_,
        )
        x_290 = (
            l_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_88 = qkv_44.reshape(1, 197, 3, 16, -1)
        qkv_44 = None
        qkv_45 = reshape_88.permute(2, 0, 3, 1, 4)
        reshape_88 = None
        unbind_22 = qkv_45.unbind(0)
        qkv_45 = None
        q_44 = unbind_22[0]
        k_44 = unbind_22[1]
        v_22 = unbind_22[2]
        unbind_22 = None
        getitem_337 = q_44[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_338 = q_44[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_44 = None
        tensor_split_44 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_44 = tensor_split_44[0]
        cos_emb_44 = tensor_split_44[1]
        tensor_split_44 = None
        mul_132 = getitem_338 * cos_emb_44
        cos_emb_44 = None
        getitem_341 = getitem_338[(Ellipsis, slice(1, None, 2))]
        neg_44 = -getitem_341
        getitem_341 = None
        getitem_342 = getitem_338[(Ellipsis, slice(None, None, 2))]
        getitem_338 = None
        stack_44 = torch.stack([neg_44, getitem_342], -1)
        neg_44 = getitem_342 = None
        reshape_89 = stack_44.reshape((1, 16, 196, 64))
        stack_44 = None
        mul_133 = reshape_89 * sin_emb_44
        reshape_89 = sin_emb_44 = None
        add_88 = mul_132 + mul_133
        mul_132 = mul_133 = None
        cat_45 = torch.cat([getitem_337, add_88], dim=2)
        getitem_337 = add_88 = None
        q_45 = cat_45.type_as(v_22)
        cat_45 = None
        getitem_343 = k_44[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_344 = k_44[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_44 = None
        tensor_split_45 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_45 = tensor_split_45[0]
        cos_emb_45 = tensor_split_45[1]
        tensor_split_45 = None
        mul_134 = getitem_344 * cos_emb_45
        cos_emb_45 = None
        getitem_347 = getitem_344[(Ellipsis, slice(1, None, 2))]
        neg_45 = -getitem_347
        getitem_347 = None
        getitem_348 = getitem_344[(Ellipsis, slice(None, None, 2))]
        getitem_344 = None
        stack_45 = torch.stack([neg_45, getitem_348], -1)
        neg_45 = getitem_348 = None
        reshape_90 = stack_45.reshape((1, 16, 196, 64))
        stack_45 = None
        mul_135 = reshape_90 * sin_emb_45
        reshape_90 = sin_emb_45 = None
        add_89 = mul_134 + mul_135
        mul_134 = mul_135 = None
        cat_46 = torch.cat([getitem_343, add_89], dim=2)
        getitem_343 = add_89 = None
        k_45 = cat_46.type_as(v_22)
        cat_46 = None
        x_291 = torch._C._nn.scaled_dot_product_attention(
            q_45, k_45, v_22, attn_mask=None, dropout_p=0.0
        )
        q_45 = k_45 = v_22 = None
        transpose_23 = x_291.transpose(1, 2)
        x_291 = None
        x_292 = transpose_23.reshape(1, 197, 1024)
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
        mul_136 = l_self_modules_blocks_modules_22_parameters_gamma_1_ * x_294
        l_self_modules_blocks_modules_22_parameters_gamma_1_ = x_294 = None
        x_295 = x_289 + mul_136
        x_289 = mul_136 = None
        x_296 = torch.nn.functional.layer_norm(
            x_295,
            (1024,),
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
        mul_137 = l_self_modules_blocks_modules_22_parameters_gamma_2_ * x_301
        l_self_modules_blocks_modules_22_parameters_gamma_2_ = x_301 = None
        x_302 = x_295 + mul_137
        x_295 = mul_137 = None
        x_303 = torch.nn.functional.layer_norm(
            x_302,
            (1024,),
            l_self_modules_blocks_modules_23_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_23_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_23_modules_norm1_parameters_bias_
        ) = None
        qkv_46 = torch._C._nn.linear(
            x_303,
            l_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_bias_,
        )
        x_303 = (
            l_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_92 = qkv_46.reshape(1, 197, 3, 16, -1)
        qkv_46 = None
        qkv_47 = reshape_92.permute(2, 0, 3, 1, 4)
        reshape_92 = None
        unbind_23 = qkv_47.unbind(0)
        qkv_47 = None
        q_46 = unbind_23[0]
        k_46 = unbind_23[1]
        v_23 = unbind_23[2]
        unbind_23 = None
        getitem_352 = q_46[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_353 = q_46[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_46 = None
        tensor_split_46 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_46 = tensor_split_46[0]
        cos_emb_46 = tensor_split_46[1]
        tensor_split_46 = None
        mul_138 = getitem_353 * cos_emb_46
        cos_emb_46 = None
        getitem_356 = getitem_353[(Ellipsis, slice(1, None, 2))]
        neg_46 = -getitem_356
        getitem_356 = None
        getitem_357 = getitem_353[(Ellipsis, slice(None, None, 2))]
        getitem_353 = None
        stack_46 = torch.stack([neg_46, getitem_357], -1)
        neg_46 = getitem_357 = None
        reshape_93 = stack_46.reshape((1, 16, 196, 64))
        stack_46 = None
        mul_139 = reshape_93 * sin_emb_46
        reshape_93 = sin_emb_46 = None
        add_92 = mul_138 + mul_139
        mul_138 = mul_139 = None
        cat_47 = torch.cat([getitem_352, add_92], dim=2)
        getitem_352 = add_92 = None
        q_47 = cat_47.type_as(v_23)
        cat_47 = None
        getitem_358 = k_46[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_359 = k_46[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_46 = None
        tensor_split_47 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        l_self_modules_rope_buffers_pos_embed_ = None
        sin_emb_47 = tensor_split_47[0]
        cos_emb_47 = tensor_split_47[1]
        tensor_split_47 = None
        mul_140 = getitem_359 * cos_emb_47
        cos_emb_47 = None
        getitem_362 = getitem_359[(Ellipsis, slice(1, None, 2))]
        neg_47 = -getitem_362
        getitem_362 = None
        getitem_363 = getitem_359[(Ellipsis, slice(None, None, 2))]
        getitem_359 = None
        stack_47 = torch.stack([neg_47, getitem_363], -1)
        neg_47 = getitem_363 = None
        reshape_94 = stack_47.reshape((1, 16, 196, 64))
        stack_47 = None
        mul_141 = reshape_94 * sin_emb_47
        reshape_94 = sin_emb_47 = None
        add_93 = mul_140 + mul_141
        mul_140 = mul_141 = None
        cat_48 = torch.cat([getitem_358, add_93], dim=2)
        getitem_358 = add_93 = None
        k_47 = cat_48.type_as(v_23)
        cat_48 = None
        x_304 = torch._C._nn.scaled_dot_product_attention(
            q_47, k_47, v_23, attn_mask=None, dropout_p=0.0
        )
        q_47 = k_47 = v_23 = None
        transpose_24 = x_304.transpose(1, 2)
        x_304 = None
        x_305 = transpose_24.reshape(1, 197, 1024)
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
        mul_142 = l_self_modules_blocks_modules_23_parameters_gamma_1_ * x_307
        l_self_modules_blocks_modules_23_parameters_gamma_1_ = x_307 = None
        x_308 = x_302 + mul_142
        x_302 = mul_142 = None
        x_309 = torch.nn.functional.layer_norm(
            x_308,
            (1024,),
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
        mul_143 = l_self_modules_blocks_modules_23_parameters_gamma_2_ * x_314
        l_self_modules_blocks_modules_23_parameters_gamma_2_ = x_314 = None
        x_315 = x_308 + mul_143
        x_308 = mul_143 = None
        x_316 = torch.nn.functional.layer_norm(
            x_315,
            (1024,),
            l_self_modules_norm_parameters_weight_,
            l_self_modules_norm_parameters_bias_,
            1e-06,
        )
        x_315 = (
            l_self_modules_norm_parameters_weight_
        ) = l_self_modules_norm_parameters_bias_ = None
        x_317 = x_316[(slice(None, None, None), 0)]
        x_316 = None
        x_318 = torch.nn.functional.dropout(x_317, 0.0, False, False)
        x_317 = None
        x_319 = torch._C._nn.linear(
            x_318,
            l_self_modules_head_parameters_weight_,
            l_self_modules_head_parameters_bias_,
        )
        x_318 = (
            l_self_modules_head_parameters_weight_
        ) = l_self_modules_head_parameters_bias_ = None
        return (x_319,)
