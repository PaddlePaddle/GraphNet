import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_x_: torch.Tensor,
        L_self_modules_patch_embed_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_patch_embed_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_parameters_pos_embed_: torch.nn.parameter.Parameter,
        L_self_modules_rope_buffers_t_x_: torch.Tensor,
        L_self_modules_rope_buffers_t_y_: torch.Tensor,
        L_self_modules_rope_parameters_freqs_: torch.nn.parameter.Parameter,
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
        l_self_parameters_pos_embed_ = L_self_parameters_pos_embed_
        l_self_modules_rope_buffers_t_x_ = L_self_modules_rope_buffers_t_x_
        l_self_modules_rope_buffers_t_y_ = L_self_modules_rope_buffers_t_y_
        l_self_modules_rope_parameters_freqs_ = L_self_modules_rope_parameters_freqs_
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
        freqs = l_self_modules_rope_parameters_freqs_.float()
        l_self_modules_rope_parameters_freqs_ = None
        unsqueeze = l_self_modules_rope_buffers_t_x_.unsqueeze(-1)
        l_self_modules_rope_buffers_t_x_ = None
        getitem_4 = freqs[0]
        unsqueeze_1 = getitem_4.unsqueeze(-2)
        getitem_4 = None
        freqs_x = unsqueeze @ unsqueeze_1
        unsqueeze = unsqueeze_1 = None
        unsqueeze_2 = l_self_modules_rope_buffers_t_y_.unsqueeze(-1)
        l_self_modules_rope_buffers_t_y_ = None
        getitem_5 = freqs[1]
        freqs = None
        unsqueeze_3 = getitem_5.unsqueeze(-2)
        getitem_5 = None
        freqs_y = unsqueeze_2 @ unsqueeze_3
        unsqueeze_2 = unsqueeze_3 = None
        combined = freqs_x + freqs_y
        freqs_x = freqs_y = None
        sin = torch.sin(combined)
        sin_emb = sin.repeat_interleave(2, -1)
        sin = None
        cos = torch.cos(combined)
        combined = None
        cos_emb = cos.repeat_interleave(2, -1)
        cos = None
        rope_embeds = torch.cat([sin_emb, cos_emb], dim=-1)
        sin_emb = cos_emb = None
        rot_pos_embed = rope_embeds.to(torch.float32)
        rope_embeds = None
        expand = l_self_parameters_cls_token_.expand(1, -1, -1)
        l_self_parameters_cls_token_ = None
        x_2 = x_1 + l_self_parameters_pos_embed_
        x_1 = l_self_parameters_pos_embed_ = None
        x_3 = torch.cat([expand, x_2], dim=1)
        expand = x_2 = None
        x_4 = torch.nn.functional.dropout(x_3, 0.0, False, False)
        x_3 = None
        getitem_6 = rot_pos_embed[0]
        x_5 = torch.nn.functional.layer_norm(
            x_4,
            (1024,),
            l_self_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_0_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_0_modules_norm1_parameters_bias_
        ) = None
        qkv = torch._C._nn.linear(
            x_5,
            l_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_,
        )
        x_5 = (
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
        getitem_10 = q[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_11 = q[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q = None
        tensor_split = getitem_6.tensor_split(2, -1)
        sin_emb_1 = tensor_split[0]
        cos_emb_1 = tensor_split[1]
        tensor_split = None
        mul = getitem_11 * cos_emb_1
        cos_emb_1 = None
        getitem_14 = getitem_11[(Ellipsis, slice(1, None, 2))]
        neg = -getitem_14
        getitem_14 = None
        getitem_15 = getitem_11[(Ellipsis, slice(None, None, 2))]
        getitem_11 = None
        stack = torch.stack([neg, getitem_15], -1)
        neg = getitem_15 = None
        reshape_1 = stack.reshape((1, 16, 196, 64))
        stack = None
        mul_1 = reshape_1 * sin_emb_1
        reshape_1 = sin_emb_1 = None
        add_2 = mul + mul_1
        mul = mul_1 = None
        cat_2 = torch.cat([getitem_10, add_2], dim=2)
        getitem_10 = add_2 = None
        q_1 = cat_2.type_as(v)
        cat_2 = None
        getitem_16 = k[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_17 = k[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k = None
        tensor_split_1 = getitem_6.tensor_split(2, -1)
        getitem_6 = None
        sin_emb_2 = tensor_split_1[0]
        cos_emb_2 = tensor_split_1[1]
        tensor_split_1 = None
        mul_2 = getitem_17 * cos_emb_2
        cos_emb_2 = None
        getitem_20 = getitem_17[(Ellipsis, slice(1, None, 2))]
        neg_1 = -getitem_20
        getitem_20 = None
        getitem_21 = getitem_17[(Ellipsis, slice(None, None, 2))]
        getitem_17 = None
        stack_1 = torch.stack([neg_1, getitem_21], -1)
        neg_1 = getitem_21 = None
        reshape_2 = stack_1.reshape((1, 16, 196, 64))
        stack_1 = None
        mul_3 = reshape_2 * sin_emb_2
        reshape_2 = sin_emb_2 = None
        add_3 = mul_2 + mul_3
        mul_2 = mul_3 = None
        cat_3 = torch.cat([getitem_16, add_3], dim=2)
        getitem_16 = add_3 = None
        k_1 = cat_3.type_as(v)
        cat_3 = None
        x_6 = torch._C._nn.scaled_dot_product_attention(
            q_1, k_1, v, attn_mask=None, dropout_p=0.0
        )
        q_1 = k_1 = v = None
        transpose_1 = x_6.transpose(1, 2)
        x_6 = None
        x_7 = transpose_1.reshape(1, 197, 1024)
        transpose_1 = None
        x_8 = torch._C._nn.linear(
            x_7,
            l_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_7 = (
            l_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_9 = torch.nn.functional.dropout(x_8, 0.0, False, False)
        x_8 = None
        mul_4 = l_self_modules_blocks_modules_0_parameters_gamma_1_ * x_9
        l_self_modules_blocks_modules_0_parameters_gamma_1_ = x_9 = None
        x_10 = x_4 + mul_4
        x_4 = mul_4 = None
        x_11 = torch.nn.functional.layer_norm(
            x_10,
            (1024,),
            l_self_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_0_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_0_modules_norm2_parameters_bias_
        ) = None
        x_12 = torch._C._nn.linear(
            x_11,
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_11 = (
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_13 = torch._C._nn.gelu(x_12, approximate="none")
        x_12 = None
        x_14 = torch.nn.functional.dropout(x_13, 0.0, False, False)
        x_13 = None
        x_15 = torch._C._nn.linear(
            x_14,
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_14 = (
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_16 = torch.nn.functional.dropout(x_15, 0.0, False, False)
        x_15 = None
        mul_5 = l_self_modules_blocks_modules_0_parameters_gamma_2_ * x_16
        l_self_modules_blocks_modules_0_parameters_gamma_2_ = x_16 = None
        x_17 = x_10 + mul_5
        x_10 = mul_5 = None
        getitem_22 = rot_pos_embed[1]
        x_18 = torch.nn.functional.layer_norm(
            x_17,
            (1024,),
            l_self_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_1_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_1_modules_norm1_parameters_bias_
        ) = None
        qkv_2 = torch._C._nn.linear(
            x_18,
            l_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_,
        )
        x_18 = (
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
        getitem_26 = q_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_27 = q_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_2 = None
        tensor_split_2 = getitem_22.tensor_split(2, -1)
        sin_emb_3 = tensor_split_2[0]
        cos_emb_3 = tensor_split_2[1]
        tensor_split_2 = None
        mul_6 = getitem_27 * cos_emb_3
        cos_emb_3 = None
        getitem_30 = getitem_27[(Ellipsis, slice(1, None, 2))]
        neg_2 = -getitem_30
        getitem_30 = None
        getitem_31 = getitem_27[(Ellipsis, slice(None, None, 2))]
        getitem_27 = None
        stack_2 = torch.stack([neg_2, getitem_31], -1)
        neg_2 = getitem_31 = None
        reshape_5 = stack_2.reshape((1, 16, 196, 64))
        stack_2 = None
        mul_7 = reshape_5 * sin_emb_3
        reshape_5 = sin_emb_3 = None
        add_6 = mul_6 + mul_7
        mul_6 = mul_7 = None
        cat_4 = torch.cat([getitem_26, add_6], dim=2)
        getitem_26 = add_6 = None
        q_3 = cat_4.type_as(v_1)
        cat_4 = None
        getitem_32 = k_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_33 = k_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_2 = None
        tensor_split_3 = getitem_22.tensor_split(2, -1)
        getitem_22 = None
        sin_emb_4 = tensor_split_3[0]
        cos_emb_4 = tensor_split_3[1]
        tensor_split_3 = None
        mul_8 = getitem_33 * cos_emb_4
        cos_emb_4 = None
        getitem_36 = getitem_33[(Ellipsis, slice(1, None, 2))]
        neg_3 = -getitem_36
        getitem_36 = None
        getitem_37 = getitem_33[(Ellipsis, slice(None, None, 2))]
        getitem_33 = None
        stack_3 = torch.stack([neg_3, getitem_37], -1)
        neg_3 = getitem_37 = None
        reshape_6 = stack_3.reshape((1, 16, 196, 64))
        stack_3 = None
        mul_9 = reshape_6 * sin_emb_4
        reshape_6 = sin_emb_4 = None
        add_7 = mul_8 + mul_9
        mul_8 = mul_9 = None
        cat_5 = torch.cat([getitem_32, add_7], dim=2)
        getitem_32 = add_7 = None
        k_3 = cat_5.type_as(v_1)
        cat_5 = None
        x_19 = torch._C._nn.scaled_dot_product_attention(
            q_3, k_3, v_1, attn_mask=None, dropout_p=0.0
        )
        q_3 = k_3 = v_1 = None
        transpose_2 = x_19.transpose(1, 2)
        x_19 = None
        x_20 = transpose_2.reshape(1, 197, 1024)
        transpose_2 = None
        x_21 = torch._C._nn.linear(
            x_20,
            l_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_20 = (
            l_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_22 = torch.nn.functional.dropout(x_21, 0.0, False, False)
        x_21 = None
        mul_10 = l_self_modules_blocks_modules_1_parameters_gamma_1_ * x_22
        l_self_modules_blocks_modules_1_parameters_gamma_1_ = x_22 = None
        x_23 = x_17 + mul_10
        x_17 = mul_10 = None
        x_24 = torch.nn.functional.layer_norm(
            x_23,
            (1024,),
            l_self_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_1_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_1_modules_norm2_parameters_bias_
        ) = None
        x_25 = torch._C._nn.linear(
            x_24,
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_24 = (
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_26 = torch._C._nn.gelu(x_25, approximate="none")
        x_25 = None
        x_27 = torch.nn.functional.dropout(x_26, 0.0, False, False)
        x_26 = None
        x_28 = torch._C._nn.linear(
            x_27,
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_27 = (
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_29 = torch.nn.functional.dropout(x_28, 0.0, False, False)
        x_28 = None
        mul_11 = l_self_modules_blocks_modules_1_parameters_gamma_2_ * x_29
        l_self_modules_blocks_modules_1_parameters_gamma_2_ = x_29 = None
        x_30 = x_23 + mul_11
        x_23 = mul_11 = None
        getitem_38 = rot_pos_embed[2]
        x_31 = torch.nn.functional.layer_norm(
            x_30,
            (1024,),
            l_self_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_2_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_2_modules_norm1_parameters_bias_
        ) = None
        qkv_4 = torch._C._nn.linear(
            x_31,
            l_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_,
        )
        x_31 = (
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
        getitem_42 = q_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_43 = q_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_4 = None
        tensor_split_4 = getitem_38.tensor_split(2, -1)
        sin_emb_5 = tensor_split_4[0]
        cos_emb_5 = tensor_split_4[1]
        tensor_split_4 = None
        mul_12 = getitem_43 * cos_emb_5
        cos_emb_5 = None
        getitem_46 = getitem_43[(Ellipsis, slice(1, None, 2))]
        neg_4 = -getitem_46
        getitem_46 = None
        getitem_47 = getitem_43[(Ellipsis, slice(None, None, 2))]
        getitem_43 = None
        stack_4 = torch.stack([neg_4, getitem_47], -1)
        neg_4 = getitem_47 = None
        reshape_9 = stack_4.reshape((1, 16, 196, 64))
        stack_4 = None
        mul_13 = reshape_9 * sin_emb_5
        reshape_9 = sin_emb_5 = None
        add_10 = mul_12 + mul_13
        mul_12 = mul_13 = None
        cat_6 = torch.cat([getitem_42, add_10], dim=2)
        getitem_42 = add_10 = None
        q_5 = cat_6.type_as(v_2)
        cat_6 = None
        getitem_48 = k_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_49 = k_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_4 = None
        tensor_split_5 = getitem_38.tensor_split(2, -1)
        getitem_38 = None
        sin_emb_6 = tensor_split_5[0]
        cos_emb_6 = tensor_split_5[1]
        tensor_split_5 = None
        mul_14 = getitem_49 * cos_emb_6
        cos_emb_6 = None
        getitem_52 = getitem_49[(Ellipsis, slice(1, None, 2))]
        neg_5 = -getitem_52
        getitem_52 = None
        getitem_53 = getitem_49[(Ellipsis, slice(None, None, 2))]
        getitem_49 = None
        stack_5 = torch.stack([neg_5, getitem_53], -1)
        neg_5 = getitem_53 = None
        reshape_10 = stack_5.reshape((1, 16, 196, 64))
        stack_5 = None
        mul_15 = reshape_10 * sin_emb_6
        reshape_10 = sin_emb_6 = None
        add_11 = mul_14 + mul_15
        mul_14 = mul_15 = None
        cat_7 = torch.cat([getitem_48, add_11], dim=2)
        getitem_48 = add_11 = None
        k_5 = cat_7.type_as(v_2)
        cat_7 = None
        x_32 = torch._C._nn.scaled_dot_product_attention(
            q_5, k_5, v_2, attn_mask=None, dropout_p=0.0
        )
        q_5 = k_5 = v_2 = None
        transpose_3 = x_32.transpose(1, 2)
        x_32 = None
        x_33 = transpose_3.reshape(1, 197, 1024)
        transpose_3 = None
        x_34 = torch._C._nn.linear(
            x_33,
            l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_,
        )
        x_33 = (
            l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_35 = torch.nn.functional.dropout(x_34, 0.0, False, False)
        x_34 = None
        mul_16 = l_self_modules_blocks_modules_2_parameters_gamma_1_ * x_35
        l_self_modules_blocks_modules_2_parameters_gamma_1_ = x_35 = None
        x_36 = x_30 + mul_16
        x_30 = mul_16 = None
        x_37 = torch.nn.functional.layer_norm(
            x_36,
            (1024,),
            l_self_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_2_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_2_modules_norm2_parameters_bias_
        ) = None
        x_38 = torch._C._nn.linear(
            x_37,
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_37 = (
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_39 = torch._C._nn.gelu(x_38, approximate="none")
        x_38 = None
        x_40 = torch.nn.functional.dropout(x_39, 0.0, False, False)
        x_39 = None
        x_41 = torch._C._nn.linear(
            x_40,
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_40 = (
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_42 = torch.nn.functional.dropout(x_41, 0.0, False, False)
        x_41 = None
        mul_17 = l_self_modules_blocks_modules_2_parameters_gamma_2_ * x_42
        l_self_modules_blocks_modules_2_parameters_gamma_2_ = x_42 = None
        x_43 = x_36 + mul_17
        x_36 = mul_17 = None
        getitem_54 = rot_pos_embed[3]
        x_44 = torch.nn.functional.layer_norm(
            x_43,
            (1024,),
            l_self_modules_blocks_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_3_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_3_modules_norm1_parameters_bias_
        ) = None
        qkv_6 = torch._C._nn.linear(
            x_44,
            l_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_bias_,
        )
        x_44 = (
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
        getitem_58 = q_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_59 = q_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_6 = None
        tensor_split_6 = getitem_54.tensor_split(2, -1)
        sin_emb_7 = tensor_split_6[0]
        cos_emb_7 = tensor_split_6[1]
        tensor_split_6 = None
        mul_18 = getitem_59 * cos_emb_7
        cos_emb_7 = None
        getitem_62 = getitem_59[(Ellipsis, slice(1, None, 2))]
        neg_6 = -getitem_62
        getitem_62 = None
        getitem_63 = getitem_59[(Ellipsis, slice(None, None, 2))]
        getitem_59 = None
        stack_6 = torch.stack([neg_6, getitem_63], -1)
        neg_6 = getitem_63 = None
        reshape_13 = stack_6.reshape((1, 16, 196, 64))
        stack_6 = None
        mul_19 = reshape_13 * sin_emb_7
        reshape_13 = sin_emb_7 = None
        add_14 = mul_18 + mul_19
        mul_18 = mul_19 = None
        cat_8 = torch.cat([getitem_58, add_14], dim=2)
        getitem_58 = add_14 = None
        q_7 = cat_8.type_as(v_3)
        cat_8 = None
        getitem_64 = k_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_65 = k_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_6 = None
        tensor_split_7 = getitem_54.tensor_split(2, -1)
        getitem_54 = None
        sin_emb_8 = tensor_split_7[0]
        cos_emb_8 = tensor_split_7[1]
        tensor_split_7 = None
        mul_20 = getitem_65 * cos_emb_8
        cos_emb_8 = None
        getitem_68 = getitem_65[(Ellipsis, slice(1, None, 2))]
        neg_7 = -getitem_68
        getitem_68 = None
        getitem_69 = getitem_65[(Ellipsis, slice(None, None, 2))]
        getitem_65 = None
        stack_7 = torch.stack([neg_7, getitem_69], -1)
        neg_7 = getitem_69 = None
        reshape_14 = stack_7.reshape((1, 16, 196, 64))
        stack_7 = None
        mul_21 = reshape_14 * sin_emb_8
        reshape_14 = sin_emb_8 = None
        add_15 = mul_20 + mul_21
        mul_20 = mul_21 = None
        cat_9 = torch.cat([getitem_64, add_15], dim=2)
        getitem_64 = add_15 = None
        k_7 = cat_9.type_as(v_3)
        cat_9 = None
        x_45 = torch._C._nn.scaled_dot_product_attention(
            q_7, k_7, v_3, attn_mask=None, dropout_p=0.0
        )
        q_7 = k_7 = v_3 = None
        transpose_4 = x_45.transpose(1, 2)
        x_45 = None
        x_46 = transpose_4.reshape(1, 197, 1024)
        transpose_4 = None
        x_47 = torch._C._nn.linear(
            x_46,
            l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_,
        )
        x_46 = (
            l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_48 = torch.nn.functional.dropout(x_47, 0.0, False, False)
        x_47 = None
        mul_22 = l_self_modules_blocks_modules_3_parameters_gamma_1_ * x_48
        l_self_modules_blocks_modules_3_parameters_gamma_1_ = x_48 = None
        x_49 = x_43 + mul_22
        x_43 = mul_22 = None
        x_50 = torch.nn.functional.layer_norm(
            x_49,
            (1024,),
            l_self_modules_blocks_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_3_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_3_modules_norm2_parameters_bias_
        ) = None
        x_51 = torch._C._nn.linear(
            x_50,
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_50 = (
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_52 = torch._C._nn.gelu(x_51, approximate="none")
        x_51 = None
        x_53 = torch.nn.functional.dropout(x_52, 0.0, False, False)
        x_52 = None
        x_54 = torch._C._nn.linear(
            x_53,
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_53 = (
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_55 = torch.nn.functional.dropout(x_54, 0.0, False, False)
        x_54 = None
        mul_23 = l_self_modules_blocks_modules_3_parameters_gamma_2_ * x_55
        l_self_modules_blocks_modules_3_parameters_gamma_2_ = x_55 = None
        x_56 = x_49 + mul_23
        x_49 = mul_23 = None
        getitem_70 = rot_pos_embed[4]
        x_57 = torch.nn.functional.layer_norm(
            x_56,
            (1024,),
            l_self_modules_blocks_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_4_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_4_modules_norm1_parameters_bias_
        ) = None
        qkv_8 = torch._C._nn.linear(
            x_57,
            l_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_bias_,
        )
        x_57 = (
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
        getitem_74 = q_8[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_75 = q_8[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_8 = None
        tensor_split_8 = getitem_70.tensor_split(2, -1)
        sin_emb_9 = tensor_split_8[0]
        cos_emb_9 = tensor_split_8[1]
        tensor_split_8 = None
        mul_24 = getitem_75 * cos_emb_9
        cos_emb_9 = None
        getitem_78 = getitem_75[(Ellipsis, slice(1, None, 2))]
        neg_8 = -getitem_78
        getitem_78 = None
        getitem_79 = getitem_75[(Ellipsis, slice(None, None, 2))]
        getitem_75 = None
        stack_8 = torch.stack([neg_8, getitem_79], -1)
        neg_8 = getitem_79 = None
        reshape_17 = stack_8.reshape((1, 16, 196, 64))
        stack_8 = None
        mul_25 = reshape_17 * sin_emb_9
        reshape_17 = sin_emb_9 = None
        add_18 = mul_24 + mul_25
        mul_24 = mul_25 = None
        cat_10 = torch.cat([getitem_74, add_18], dim=2)
        getitem_74 = add_18 = None
        q_9 = cat_10.type_as(v_4)
        cat_10 = None
        getitem_80 = k_8[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_81 = k_8[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_8 = None
        tensor_split_9 = getitem_70.tensor_split(2, -1)
        getitem_70 = None
        sin_emb_10 = tensor_split_9[0]
        cos_emb_10 = tensor_split_9[1]
        tensor_split_9 = None
        mul_26 = getitem_81 * cos_emb_10
        cos_emb_10 = None
        getitem_84 = getitem_81[(Ellipsis, slice(1, None, 2))]
        neg_9 = -getitem_84
        getitem_84 = None
        getitem_85 = getitem_81[(Ellipsis, slice(None, None, 2))]
        getitem_81 = None
        stack_9 = torch.stack([neg_9, getitem_85], -1)
        neg_9 = getitem_85 = None
        reshape_18 = stack_9.reshape((1, 16, 196, 64))
        stack_9 = None
        mul_27 = reshape_18 * sin_emb_10
        reshape_18 = sin_emb_10 = None
        add_19 = mul_26 + mul_27
        mul_26 = mul_27 = None
        cat_11 = torch.cat([getitem_80, add_19], dim=2)
        getitem_80 = add_19 = None
        k_9 = cat_11.type_as(v_4)
        cat_11 = None
        x_58 = torch._C._nn.scaled_dot_product_attention(
            q_9, k_9, v_4, attn_mask=None, dropout_p=0.0
        )
        q_9 = k_9 = v_4 = None
        transpose_5 = x_58.transpose(1, 2)
        x_58 = None
        x_59 = transpose_5.reshape(1, 197, 1024)
        transpose_5 = None
        x_60 = torch._C._nn.linear(
            x_59,
            l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_,
        )
        x_59 = (
            l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_61 = torch.nn.functional.dropout(x_60, 0.0, False, False)
        x_60 = None
        mul_28 = l_self_modules_blocks_modules_4_parameters_gamma_1_ * x_61
        l_self_modules_blocks_modules_4_parameters_gamma_1_ = x_61 = None
        x_62 = x_56 + mul_28
        x_56 = mul_28 = None
        x_63 = torch.nn.functional.layer_norm(
            x_62,
            (1024,),
            l_self_modules_blocks_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_4_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_4_modules_norm2_parameters_bias_
        ) = None
        x_64 = torch._C._nn.linear(
            x_63,
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_63 = (
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_65 = torch._C._nn.gelu(x_64, approximate="none")
        x_64 = None
        x_66 = torch.nn.functional.dropout(x_65, 0.0, False, False)
        x_65 = None
        x_67 = torch._C._nn.linear(
            x_66,
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_66 = (
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_68 = torch.nn.functional.dropout(x_67, 0.0, False, False)
        x_67 = None
        mul_29 = l_self_modules_blocks_modules_4_parameters_gamma_2_ * x_68
        l_self_modules_blocks_modules_4_parameters_gamma_2_ = x_68 = None
        x_69 = x_62 + mul_29
        x_62 = mul_29 = None
        getitem_86 = rot_pos_embed[5]
        x_70 = torch.nn.functional.layer_norm(
            x_69,
            (1024,),
            l_self_modules_blocks_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_5_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_5_modules_norm1_parameters_bias_
        ) = None
        qkv_10 = torch._C._nn.linear(
            x_70,
            l_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_bias_,
        )
        x_70 = (
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
        getitem_90 = q_10[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_91 = q_10[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_10 = None
        tensor_split_10 = getitem_86.tensor_split(2, -1)
        sin_emb_11 = tensor_split_10[0]
        cos_emb_11 = tensor_split_10[1]
        tensor_split_10 = None
        mul_30 = getitem_91 * cos_emb_11
        cos_emb_11 = None
        getitem_94 = getitem_91[(Ellipsis, slice(1, None, 2))]
        neg_10 = -getitem_94
        getitem_94 = None
        getitem_95 = getitem_91[(Ellipsis, slice(None, None, 2))]
        getitem_91 = None
        stack_10 = torch.stack([neg_10, getitem_95], -1)
        neg_10 = getitem_95 = None
        reshape_21 = stack_10.reshape((1, 16, 196, 64))
        stack_10 = None
        mul_31 = reshape_21 * sin_emb_11
        reshape_21 = sin_emb_11 = None
        add_22 = mul_30 + mul_31
        mul_30 = mul_31 = None
        cat_12 = torch.cat([getitem_90, add_22], dim=2)
        getitem_90 = add_22 = None
        q_11 = cat_12.type_as(v_5)
        cat_12 = None
        getitem_96 = k_10[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_97 = k_10[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_10 = None
        tensor_split_11 = getitem_86.tensor_split(2, -1)
        getitem_86 = None
        sin_emb_12 = tensor_split_11[0]
        cos_emb_12 = tensor_split_11[1]
        tensor_split_11 = None
        mul_32 = getitem_97 * cos_emb_12
        cos_emb_12 = None
        getitem_100 = getitem_97[(Ellipsis, slice(1, None, 2))]
        neg_11 = -getitem_100
        getitem_100 = None
        getitem_101 = getitem_97[(Ellipsis, slice(None, None, 2))]
        getitem_97 = None
        stack_11 = torch.stack([neg_11, getitem_101], -1)
        neg_11 = getitem_101 = None
        reshape_22 = stack_11.reshape((1, 16, 196, 64))
        stack_11 = None
        mul_33 = reshape_22 * sin_emb_12
        reshape_22 = sin_emb_12 = None
        add_23 = mul_32 + mul_33
        mul_32 = mul_33 = None
        cat_13 = torch.cat([getitem_96, add_23], dim=2)
        getitem_96 = add_23 = None
        k_11 = cat_13.type_as(v_5)
        cat_13 = None
        x_71 = torch._C._nn.scaled_dot_product_attention(
            q_11, k_11, v_5, attn_mask=None, dropout_p=0.0
        )
        q_11 = k_11 = v_5 = None
        transpose_6 = x_71.transpose(1, 2)
        x_71 = None
        x_72 = transpose_6.reshape(1, 197, 1024)
        transpose_6 = None
        x_73 = torch._C._nn.linear(
            x_72,
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_,
        )
        x_72 = (
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_74 = torch.nn.functional.dropout(x_73, 0.0, False, False)
        x_73 = None
        mul_34 = l_self_modules_blocks_modules_5_parameters_gamma_1_ * x_74
        l_self_modules_blocks_modules_5_parameters_gamma_1_ = x_74 = None
        x_75 = x_69 + mul_34
        x_69 = mul_34 = None
        x_76 = torch.nn.functional.layer_norm(
            x_75,
            (1024,),
            l_self_modules_blocks_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_5_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_5_modules_norm2_parameters_bias_
        ) = None
        x_77 = torch._C._nn.linear(
            x_76,
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_76 = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_78 = torch._C._nn.gelu(x_77, approximate="none")
        x_77 = None
        x_79 = torch.nn.functional.dropout(x_78, 0.0, False, False)
        x_78 = None
        x_80 = torch._C._nn.linear(
            x_79,
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_79 = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_81 = torch.nn.functional.dropout(x_80, 0.0, False, False)
        x_80 = None
        mul_35 = l_self_modules_blocks_modules_5_parameters_gamma_2_ * x_81
        l_self_modules_blocks_modules_5_parameters_gamma_2_ = x_81 = None
        x_82 = x_75 + mul_35
        x_75 = mul_35 = None
        getitem_102 = rot_pos_embed[6]
        x_83 = torch.nn.functional.layer_norm(
            x_82,
            (1024,),
            l_self_modules_blocks_modules_6_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_6_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_6_modules_norm1_parameters_bias_
        ) = None
        qkv_12 = torch._C._nn.linear(
            x_83,
            l_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_bias_,
        )
        x_83 = (
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
        getitem_106 = q_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_107 = q_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_12 = None
        tensor_split_12 = getitem_102.tensor_split(2, -1)
        sin_emb_13 = tensor_split_12[0]
        cos_emb_13 = tensor_split_12[1]
        tensor_split_12 = None
        mul_36 = getitem_107 * cos_emb_13
        cos_emb_13 = None
        getitem_110 = getitem_107[(Ellipsis, slice(1, None, 2))]
        neg_12 = -getitem_110
        getitem_110 = None
        getitem_111 = getitem_107[(Ellipsis, slice(None, None, 2))]
        getitem_107 = None
        stack_12 = torch.stack([neg_12, getitem_111], -1)
        neg_12 = getitem_111 = None
        reshape_25 = stack_12.reshape((1, 16, 196, 64))
        stack_12 = None
        mul_37 = reshape_25 * sin_emb_13
        reshape_25 = sin_emb_13 = None
        add_26 = mul_36 + mul_37
        mul_36 = mul_37 = None
        cat_14 = torch.cat([getitem_106, add_26], dim=2)
        getitem_106 = add_26 = None
        q_13 = cat_14.type_as(v_6)
        cat_14 = None
        getitem_112 = k_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_113 = k_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_12 = None
        tensor_split_13 = getitem_102.tensor_split(2, -1)
        getitem_102 = None
        sin_emb_14 = tensor_split_13[0]
        cos_emb_14 = tensor_split_13[1]
        tensor_split_13 = None
        mul_38 = getitem_113 * cos_emb_14
        cos_emb_14 = None
        getitem_116 = getitem_113[(Ellipsis, slice(1, None, 2))]
        neg_13 = -getitem_116
        getitem_116 = None
        getitem_117 = getitem_113[(Ellipsis, slice(None, None, 2))]
        getitem_113 = None
        stack_13 = torch.stack([neg_13, getitem_117], -1)
        neg_13 = getitem_117 = None
        reshape_26 = stack_13.reshape((1, 16, 196, 64))
        stack_13 = None
        mul_39 = reshape_26 * sin_emb_14
        reshape_26 = sin_emb_14 = None
        add_27 = mul_38 + mul_39
        mul_38 = mul_39 = None
        cat_15 = torch.cat([getitem_112, add_27], dim=2)
        getitem_112 = add_27 = None
        k_13 = cat_15.type_as(v_6)
        cat_15 = None
        x_84 = torch._C._nn.scaled_dot_product_attention(
            q_13, k_13, v_6, attn_mask=None, dropout_p=0.0
        )
        q_13 = k_13 = v_6 = None
        transpose_7 = x_84.transpose(1, 2)
        x_84 = None
        x_85 = transpose_7.reshape(1, 197, 1024)
        transpose_7 = None
        x_86 = torch._C._nn.linear(
            x_85,
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_,
        )
        x_85 = (
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_87 = torch.nn.functional.dropout(x_86, 0.0, False, False)
        x_86 = None
        mul_40 = l_self_modules_blocks_modules_6_parameters_gamma_1_ * x_87
        l_self_modules_blocks_modules_6_parameters_gamma_1_ = x_87 = None
        x_88 = x_82 + mul_40
        x_82 = mul_40 = None
        x_89 = torch.nn.functional.layer_norm(
            x_88,
            (1024,),
            l_self_modules_blocks_modules_6_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_6_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_6_modules_norm2_parameters_bias_
        ) = None
        x_90 = torch._C._nn.linear(
            x_89,
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_89 = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_91 = torch._C._nn.gelu(x_90, approximate="none")
        x_90 = None
        x_92 = torch.nn.functional.dropout(x_91, 0.0, False, False)
        x_91 = None
        x_93 = torch._C._nn.linear(
            x_92,
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_92 = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_94 = torch.nn.functional.dropout(x_93, 0.0, False, False)
        x_93 = None
        mul_41 = l_self_modules_blocks_modules_6_parameters_gamma_2_ * x_94
        l_self_modules_blocks_modules_6_parameters_gamma_2_ = x_94 = None
        x_95 = x_88 + mul_41
        x_88 = mul_41 = None
        getitem_118 = rot_pos_embed[7]
        x_96 = torch.nn.functional.layer_norm(
            x_95,
            (1024,),
            l_self_modules_blocks_modules_7_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_7_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_7_modules_norm1_parameters_bias_
        ) = None
        qkv_14 = torch._C._nn.linear(
            x_96,
            l_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_bias_,
        )
        x_96 = (
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
        getitem_122 = q_14[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_123 = q_14[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_14 = None
        tensor_split_14 = getitem_118.tensor_split(2, -1)
        sin_emb_15 = tensor_split_14[0]
        cos_emb_15 = tensor_split_14[1]
        tensor_split_14 = None
        mul_42 = getitem_123 * cos_emb_15
        cos_emb_15 = None
        getitem_126 = getitem_123[(Ellipsis, slice(1, None, 2))]
        neg_14 = -getitem_126
        getitem_126 = None
        getitem_127 = getitem_123[(Ellipsis, slice(None, None, 2))]
        getitem_123 = None
        stack_14 = torch.stack([neg_14, getitem_127], -1)
        neg_14 = getitem_127 = None
        reshape_29 = stack_14.reshape((1, 16, 196, 64))
        stack_14 = None
        mul_43 = reshape_29 * sin_emb_15
        reshape_29 = sin_emb_15 = None
        add_30 = mul_42 + mul_43
        mul_42 = mul_43 = None
        cat_16 = torch.cat([getitem_122, add_30], dim=2)
        getitem_122 = add_30 = None
        q_15 = cat_16.type_as(v_7)
        cat_16 = None
        getitem_128 = k_14[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_129 = k_14[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_14 = None
        tensor_split_15 = getitem_118.tensor_split(2, -1)
        getitem_118 = None
        sin_emb_16 = tensor_split_15[0]
        cos_emb_16 = tensor_split_15[1]
        tensor_split_15 = None
        mul_44 = getitem_129 * cos_emb_16
        cos_emb_16 = None
        getitem_132 = getitem_129[(Ellipsis, slice(1, None, 2))]
        neg_15 = -getitem_132
        getitem_132 = None
        getitem_133 = getitem_129[(Ellipsis, slice(None, None, 2))]
        getitem_129 = None
        stack_15 = torch.stack([neg_15, getitem_133], -1)
        neg_15 = getitem_133 = None
        reshape_30 = stack_15.reshape((1, 16, 196, 64))
        stack_15 = None
        mul_45 = reshape_30 * sin_emb_16
        reshape_30 = sin_emb_16 = None
        add_31 = mul_44 + mul_45
        mul_44 = mul_45 = None
        cat_17 = torch.cat([getitem_128, add_31], dim=2)
        getitem_128 = add_31 = None
        k_15 = cat_17.type_as(v_7)
        cat_17 = None
        x_97 = torch._C._nn.scaled_dot_product_attention(
            q_15, k_15, v_7, attn_mask=None, dropout_p=0.0
        )
        q_15 = k_15 = v_7 = None
        transpose_8 = x_97.transpose(1, 2)
        x_97 = None
        x_98 = transpose_8.reshape(1, 197, 1024)
        transpose_8 = None
        x_99 = torch._C._nn.linear(
            x_98,
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_,
        )
        x_98 = (
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_100 = torch.nn.functional.dropout(x_99, 0.0, False, False)
        x_99 = None
        mul_46 = l_self_modules_blocks_modules_7_parameters_gamma_1_ * x_100
        l_self_modules_blocks_modules_7_parameters_gamma_1_ = x_100 = None
        x_101 = x_95 + mul_46
        x_95 = mul_46 = None
        x_102 = torch.nn.functional.layer_norm(
            x_101,
            (1024,),
            l_self_modules_blocks_modules_7_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_7_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_7_modules_norm2_parameters_bias_
        ) = None
        x_103 = torch._C._nn.linear(
            x_102,
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_102 = (
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_104 = torch._C._nn.gelu(x_103, approximate="none")
        x_103 = None
        x_105 = torch.nn.functional.dropout(x_104, 0.0, False, False)
        x_104 = None
        x_106 = torch._C._nn.linear(
            x_105,
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_105 = (
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_107 = torch.nn.functional.dropout(x_106, 0.0, False, False)
        x_106 = None
        mul_47 = l_self_modules_blocks_modules_7_parameters_gamma_2_ * x_107
        l_self_modules_blocks_modules_7_parameters_gamma_2_ = x_107 = None
        x_108 = x_101 + mul_47
        x_101 = mul_47 = None
        getitem_134 = rot_pos_embed[8]
        x_109 = torch.nn.functional.layer_norm(
            x_108,
            (1024,),
            l_self_modules_blocks_modules_8_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_8_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_8_modules_norm1_parameters_bias_
        ) = None
        qkv_16 = torch._C._nn.linear(
            x_109,
            l_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_bias_,
        )
        x_109 = (
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
        getitem_138 = q_16[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_139 = q_16[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_16 = None
        tensor_split_16 = getitem_134.tensor_split(2, -1)
        sin_emb_17 = tensor_split_16[0]
        cos_emb_17 = tensor_split_16[1]
        tensor_split_16 = None
        mul_48 = getitem_139 * cos_emb_17
        cos_emb_17 = None
        getitem_142 = getitem_139[(Ellipsis, slice(1, None, 2))]
        neg_16 = -getitem_142
        getitem_142 = None
        getitem_143 = getitem_139[(Ellipsis, slice(None, None, 2))]
        getitem_139 = None
        stack_16 = torch.stack([neg_16, getitem_143], -1)
        neg_16 = getitem_143 = None
        reshape_33 = stack_16.reshape((1, 16, 196, 64))
        stack_16 = None
        mul_49 = reshape_33 * sin_emb_17
        reshape_33 = sin_emb_17 = None
        add_34 = mul_48 + mul_49
        mul_48 = mul_49 = None
        cat_18 = torch.cat([getitem_138, add_34], dim=2)
        getitem_138 = add_34 = None
        q_17 = cat_18.type_as(v_8)
        cat_18 = None
        getitem_144 = k_16[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_145 = k_16[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_16 = None
        tensor_split_17 = getitem_134.tensor_split(2, -1)
        getitem_134 = None
        sin_emb_18 = tensor_split_17[0]
        cos_emb_18 = tensor_split_17[1]
        tensor_split_17 = None
        mul_50 = getitem_145 * cos_emb_18
        cos_emb_18 = None
        getitem_148 = getitem_145[(Ellipsis, slice(1, None, 2))]
        neg_17 = -getitem_148
        getitem_148 = None
        getitem_149 = getitem_145[(Ellipsis, slice(None, None, 2))]
        getitem_145 = None
        stack_17 = torch.stack([neg_17, getitem_149], -1)
        neg_17 = getitem_149 = None
        reshape_34 = stack_17.reshape((1, 16, 196, 64))
        stack_17 = None
        mul_51 = reshape_34 * sin_emb_18
        reshape_34 = sin_emb_18 = None
        add_35 = mul_50 + mul_51
        mul_50 = mul_51 = None
        cat_19 = torch.cat([getitem_144, add_35], dim=2)
        getitem_144 = add_35 = None
        k_17 = cat_19.type_as(v_8)
        cat_19 = None
        x_110 = torch._C._nn.scaled_dot_product_attention(
            q_17, k_17, v_8, attn_mask=None, dropout_p=0.0
        )
        q_17 = k_17 = v_8 = None
        transpose_9 = x_110.transpose(1, 2)
        x_110 = None
        x_111 = transpose_9.reshape(1, 197, 1024)
        transpose_9 = None
        x_112 = torch._C._nn.linear(
            x_111,
            l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_,
        )
        x_111 = (
            l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_113 = torch.nn.functional.dropout(x_112, 0.0, False, False)
        x_112 = None
        mul_52 = l_self_modules_blocks_modules_8_parameters_gamma_1_ * x_113
        l_self_modules_blocks_modules_8_parameters_gamma_1_ = x_113 = None
        x_114 = x_108 + mul_52
        x_108 = mul_52 = None
        x_115 = torch.nn.functional.layer_norm(
            x_114,
            (1024,),
            l_self_modules_blocks_modules_8_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_8_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_8_modules_norm2_parameters_bias_
        ) = None
        x_116 = torch._C._nn.linear(
            x_115,
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_115 = (
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_117 = torch._C._nn.gelu(x_116, approximate="none")
        x_116 = None
        x_118 = torch.nn.functional.dropout(x_117, 0.0, False, False)
        x_117 = None
        x_119 = torch._C._nn.linear(
            x_118,
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_118 = (
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_120 = torch.nn.functional.dropout(x_119, 0.0, False, False)
        x_119 = None
        mul_53 = l_self_modules_blocks_modules_8_parameters_gamma_2_ * x_120
        l_self_modules_blocks_modules_8_parameters_gamma_2_ = x_120 = None
        x_121 = x_114 + mul_53
        x_114 = mul_53 = None
        getitem_150 = rot_pos_embed[9]
        x_122 = torch.nn.functional.layer_norm(
            x_121,
            (1024,),
            l_self_modules_blocks_modules_9_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_9_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_9_modules_norm1_parameters_bias_
        ) = None
        qkv_18 = torch._C._nn.linear(
            x_122,
            l_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_bias_,
        )
        x_122 = (
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
        getitem_154 = q_18[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_155 = q_18[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_18 = None
        tensor_split_18 = getitem_150.tensor_split(2, -1)
        sin_emb_19 = tensor_split_18[0]
        cos_emb_19 = tensor_split_18[1]
        tensor_split_18 = None
        mul_54 = getitem_155 * cos_emb_19
        cos_emb_19 = None
        getitem_158 = getitem_155[(Ellipsis, slice(1, None, 2))]
        neg_18 = -getitem_158
        getitem_158 = None
        getitem_159 = getitem_155[(Ellipsis, slice(None, None, 2))]
        getitem_155 = None
        stack_18 = torch.stack([neg_18, getitem_159], -1)
        neg_18 = getitem_159 = None
        reshape_37 = stack_18.reshape((1, 16, 196, 64))
        stack_18 = None
        mul_55 = reshape_37 * sin_emb_19
        reshape_37 = sin_emb_19 = None
        add_38 = mul_54 + mul_55
        mul_54 = mul_55 = None
        cat_20 = torch.cat([getitem_154, add_38], dim=2)
        getitem_154 = add_38 = None
        q_19 = cat_20.type_as(v_9)
        cat_20 = None
        getitem_160 = k_18[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_161 = k_18[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_18 = None
        tensor_split_19 = getitem_150.tensor_split(2, -1)
        getitem_150 = None
        sin_emb_20 = tensor_split_19[0]
        cos_emb_20 = tensor_split_19[1]
        tensor_split_19 = None
        mul_56 = getitem_161 * cos_emb_20
        cos_emb_20 = None
        getitem_164 = getitem_161[(Ellipsis, slice(1, None, 2))]
        neg_19 = -getitem_164
        getitem_164 = None
        getitem_165 = getitem_161[(Ellipsis, slice(None, None, 2))]
        getitem_161 = None
        stack_19 = torch.stack([neg_19, getitem_165], -1)
        neg_19 = getitem_165 = None
        reshape_38 = stack_19.reshape((1, 16, 196, 64))
        stack_19 = None
        mul_57 = reshape_38 * sin_emb_20
        reshape_38 = sin_emb_20 = None
        add_39 = mul_56 + mul_57
        mul_56 = mul_57 = None
        cat_21 = torch.cat([getitem_160, add_39], dim=2)
        getitem_160 = add_39 = None
        k_19 = cat_21.type_as(v_9)
        cat_21 = None
        x_123 = torch._C._nn.scaled_dot_product_attention(
            q_19, k_19, v_9, attn_mask=None, dropout_p=0.0
        )
        q_19 = k_19 = v_9 = None
        transpose_10 = x_123.transpose(1, 2)
        x_123 = None
        x_124 = transpose_10.reshape(1, 197, 1024)
        transpose_10 = None
        x_125 = torch._C._nn.linear(
            x_124,
            l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_,
        )
        x_124 = (
            l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_126 = torch.nn.functional.dropout(x_125, 0.0, False, False)
        x_125 = None
        mul_58 = l_self_modules_blocks_modules_9_parameters_gamma_1_ * x_126
        l_self_modules_blocks_modules_9_parameters_gamma_1_ = x_126 = None
        x_127 = x_121 + mul_58
        x_121 = mul_58 = None
        x_128 = torch.nn.functional.layer_norm(
            x_127,
            (1024,),
            l_self_modules_blocks_modules_9_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_9_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_9_modules_norm2_parameters_bias_
        ) = None
        x_129 = torch._C._nn.linear(
            x_128,
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_128 = (
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_130 = torch._C._nn.gelu(x_129, approximate="none")
        x_129 = None
        x_131 = torch.nn.functional.dropout(x_130, 0.0, False, False)
        x_130 = None
        x_132 = torch._C._nn.linear(
            x_131,
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_131 = (
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_133 = torch.nn.functional.dropout(x_132, 0.0, False, False)
        x_132 = None
        mul_59 = l_self_modules_blocks_modules_9_parameters_gamma_2_ * x_133
        l_self_modules_blocks_modules_9_parameters_gamma_2_ = x_133 = None
        x_134 = x_127 + mul_59
        x_127 = mul_59 = None
        getitem_166 = rot_pos_embed[10]
        x_135 = torch.nn.functional.layer_norm(
            x_134,
            (1024,),
            l_self_modules_blocks_modules_10_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_10_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_10_modules_norm1_parameters_bias_
        ) = None
        qkv_20 = torch._C._nn.linear(
            x_135,
            l_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_bias_,
        )
        x_135 = (
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
        getitem_170 = q_20[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_171 = q_20[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_20 = None
        tensor_split_20 = getitem_166.tensor_split(2, -1)
        sin_emb_21 = tensor_split_20[0]
        cos_emb_21 = tensor_split_20[1]
        tensor_split_20 = None
        mul_60 = getitem_171 * cos_emb_21
        cos_emb_21 = None
        getitem_174 = getitem_171[(Ellipsis, slice(1, None, 2))]
        neg_20 = -getitem_174
        getitem_174 = None
        getitem_175 = getitem_171[(Ellipsis, slice(None, None, 2))]
        getitem_171 = None
        stack_20 = torch.stack([neg_20, getitem_175], -1)
        neg_20 = getitem_175 = None
        reshape_41 = stack_20.reshape((1, 16, 196, 64))
        stack_20 = None
        mul_61 = reshape_41 * sin_emb_21
        reshape_41 = sin_emb_21 = None
        add_42 = mul_60 + mul_61
        mul_60 = mul_61 = None
        cat_22 = torch.cat([getitem_170, add_42], dim=2)
        getitem_170 = add_42 = None
        q_21 = cat_22.type_as(v_10)
        cat_22 = None
        getitem_176 = k_20[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_177 = k_20[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_20 = None
        tensor_split_21 = getitem_166.tensor_split(2, -1)
        getitem_166 = None
        sin_emb_22 = tensor_split_21[0]
        cos_emb_22 = tensor_split_21[1]
        tensor_split_21 = None
        mul_62 = getitem_177 * cos_emb_22
        cos_emb_22 = None
        getitem_180 = getitem_177[(Ellipsis, slice(1, None, 2))]
        neg_21 = -getitem_180
        getitem_180 = None
        getitem_181 = getitem_177[(Ellipsis, slice(None, None, 2))]
        getitem_177 = None
        stack_21 = torch.stack([neg_21, getitem_181], -1)
        neg_21 = getitem_181 = None
        reshape_42 = stack_21.reshape((1, 16, 196, 64))
        stack_21 = None
        mul_63 = reshape_42 * sin_emb_22
        reshape_42 = sin_emb_22 = None
        add_43 = mul_62 + mul_63
        mul_62 = mul_63 = None
        cat_23 = torch.cat([getitem_176, add_43], dim=2)
        getitem_176 = add_43 = None
        k_21 = cat_23.type_as(v_10)
        cat_23 = None
        x_136 = torch._C._nn.scaled_dot_product_attention(
            q_21, k_21, v_10, attn_mask=None, dropout_p=0.0
        )
        q_21 = k_21 = v_10 = None
        transpose_11 = x_136.transpose(1, 2)
        x_136 = None
        x_137 = transpose_11.reshape(1, 197, 1024)
        transpose_11 = None
        x_138 = torch._C._nn.linear(
            x_137,
            l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_,
        )
        x_137 = l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_139 = torch.nn.functional.dropout(x_138, 0.0, False, False)
        x_138 = None
        mul_64 = l_self_modules_blocks_modules_10_parameters_gamma_1_ * x_139
        l_self_modules_blocks_modules_10_parameters_gamma_1_ = x_139 = None
        x_140 = x_134 + mul_64
        x_134 = mul_64 = None
        x_141 = torch.nn.functional.layer_norm(
            x_140,
            (1024,),
            l_self_modules_blocks_modules_10_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_10_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_10_modules_norm2_parameters_bias_
        ) = None
        x_142 = torch._C._nn.linear(
            x_141,
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_141 = (
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_143 = torch._C._nn.gelu(x_142, approximate="none")
        x_142 = None
        x_144 = torch.nn.functional.dropout(x_143, 0.0, False, False)
        x_143 = None
        x_145 = torch._C._nn.linear(
            x_144,
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_144 = (
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_146 = torch.nn.functional.dropout(x_145, 0.0, False, False)
        x_145 = None
        mul_65 = l_self_modules_blocks_modules_10_parameters_gamma_2_ * x_146
        l_self_modules_blocks_modules_10_parameters_gamma_2_ = x_146 = None
        x_147 = x_140 + mul_65
        x_140 = mul_65 = None
        getitem_182 = rot_pos_embed[11]
        x_148 = torch.nn.functional.layer_norm(
            x_147,
            (1024,),
            l_self_modules_blocks_modules_11_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_11_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_11_modules_norm1_parameters_bias_
        ) = None
        qkv_22 = torch._C._nn.linear(
            x_148,
            l_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_bias_,
        )
        x_148 = (
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
        getitem_186 = q_22[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_187 = q_22[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_22 = None
        tensor_split_22 = getitem_182.tensor_split(2, -1)
        sin_emb_23 = tensor_split_22[0]
        cos_emb_23 = tensor_split_22[1]
        tensor_split_22 = None
        mul_66 = getitem_187 * cos_emb_23
        cos_emb_23 = None
        getitem_190 = getitem_187[(Ellipsis, slice(1, None, 2))]
        neg_22 = -getitem_190
        getitem_190 = None
        getitem_191 = getitem_187[(Ellipsis, slice(None, None, 2))]
        getitem_187 = None
        stack_22 = torch.stack([neg_22, getitem_191], -1)
        neg_22 = getitem_191 = None
        reshape_45 = stack_22.reshape((1, 16, 196, 64))
        stack_22 = None
        mul_67 = reshape_45 * sin_emb_23
        reshape_45 = sin_emb_23 = None
        add_46 = mul_66 + mul_67
        mul_66 = mul_67 = None
        cat_24 = torch.cat([getitem_186, add_46], dim=2)
        getitem_186 = add_46 = None
        q_23 = cat_24.type_as(v_11)
        cat_24 = None
        getitem_192 = k_22[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_193 = k_22[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_22 = None
        tensor_split_23 = getitem_182.tensor_split(2, -1)
        getitem_182 = None
        sin_emb_24 = tensor_split_23[0]
        cos_emb_24 = tensor_split_23[1]
        tensor_split_23 = None
        mul_68 = getitem_193 * cos_emb_24
        cos_emb_24 = None
        getitem_196 = getitem_193[(Ellipsis, slice(1, None, 2))]
        neg_23 = -getitem_196
        getitem_196 = None
        getitem_197 = getitem_193[(Ellipsis, slice(None, None, 2))]
        getitem_193 = None
        stack_23 = torch.stack([neg_23, getitem_197], -1)
        neg_23 = getitem_197 = None
        reshape_46 = stack_23.reshape((1, 16, 196, 64))
        stack_23 = None
        mul_69 = reshape_46 * sin_emb_24
        reshape_46 = sin_emb_24 = None
        add_47 = mul_68 + mul_69
        mul_68 = mul_69 = None
        cat_25 = torch.cat([getitem_192, add_47], dim=2)
        getitem_192 = add_47 = None
        k_23 = cat_25.type_as(v_11)
        cat_25 = None
        x_149 = torch._C._nn.scaled_dot_product_attention(
            q_23, k_23, v_11, attn_mask=None, dropout_p=0.0
        )
        q_23 = k_23 = v_11 = None
        transpose_12 = x_149.transpose(1, 2)
        x_149 = None
        x_150 = transpose_12.reshape(1, 197, 1024)
        transpose_12 = None
        x_151 = torch._C._nn.linear(
            x_150,
            l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_,
        )
        x_150 = l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_152 = torch.nn.functional.dropout(x_151, 0.0, False, False)
        x_151 = None
        mul_70 = l_self_modules_blocks_modules_11_parameters_gamma_1_ * x_152
        l_self_modules_blocks_modules_11_parameters_gamma_1_ = x_152 = None
        x_153 = x_147 + mul_70
        x_147 = mul_70 = None
        x_154 = torch.nn.functional.layer_norm(
            x_153,
            (1024,),
            l_self_modules_blocks_modules_11_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_11_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_11_modules_norm2_parameters_bias_
        ) = None
        x_155 = torch._C._nn.linear(
            x_154,
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_154 = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_156 = torch._C._nn.gelu(x_155, approximate="none")
        x_155 = None
        x_157 = torch.nn.functional.dropout(x_156, 0.0, False, False)
        x_156 = None
        x_158 = torch._C._nn.linear(
            x_157,
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_157 = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_159 = torch.nn.functional.dropout(x_158, 0.0, False, False)
        x_158 = None
        mul_71 = l_self_modules_blocks_modules_11_parameters_gamma_2_ * x_159
        l_self_modules_blocks_modules_11_parameters_gamma_2_ = x_159 = None
        x_160 = x_153 + mul_71
        x_153 = mul_71 = None
        getitem_198 = rot_pos_embed[12]
        x_161 = torch.nn.functional.layer_norm(
            x_160,
            (1024,),
            l_self_modules_blocks_modules_12_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_12_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_12_modules_norm1_parameters_bias_
        ) = None
        qkv_24 = torch._C._nn.linear(
            x_161,
            l_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_bias_,
        )
        x_161 = (
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
        getitem_202 = q_24[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_203 = q_24[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_24 = None
        tensor_split_24 = getitem_198.tensor_split(2, -1)
        sin_emb_25 = tensor_split_24[0]
        cos_emb_25 = tensor_split_24[1]
        tensor_split_24 = None
        mul_72 = getitem_203 * cos_emb_25
        cos_emb_25 = None
        getitem_206 = getitem_203[(Ellipsis, slice(1, None, 2))]
        neg_24 = -getitem_206
        getitem_206 = None
        getitem_207 = getitem_203[(Ellipsis, slice(None, None, 2))]
        getitem_203 = None
        stack_24 = torch.stack([neg_24, getitem_207], -1)
        neg_24 = getitem_207 = None
        reshape_49 = stack_24.reshape((1, 16, 196, 64))
        stack_24 = None
        mul_73 = reshape_49 * sin_emb_25
        reshape_49 = sin_emb_25 = None
        add_50 = mul_72 + mul_73
        mul_72 = mul_73 = None
        cat_26 = torch.cat([getitem_202, add_50], dim=2)
        getitem_202 = add_50 = None
        q_25 = cat_26.type_as(v_12)
        cat_26 = None
        getitem_208 = k_24[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_209 = k_24[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_24 = None
        tensor_split_25 = getitem_198.tensor_split(2, -1)
        getitem_198 = None
        sin_emb_26 = tensor_split_25[0]
        cos_emb_26 = tensor_split_25[1]
        tensor_split_25 = None
        mul_74 = getitem_209 * cos_emb_26
        cos_emb_26 = None
        getitem_212 = getitem_209[(Ellipsis, slice(1, None, 2))]
        neg_25 = -getitem_212
        getitem_212 = None
        getitem_213 = getitem_209[(Ellipsis, slice(None, None, 2))]
        getitem_209 = None
        stack_25 = torch.stack([neg_25, getitem_213], -1)
        neg_25 = getitem_213 = None
        reshape_50 = stack_25.reshape((1, 16, 196, 64))
        stack_25 = None
        mul_75 = reshape_50 * sin_emb_26
        reshape_50 = sin_emb_26 = None
        add_51 = mul_74 + mul_75
        mul_74 = mul_75 = None
        cat_27 = torch.cat([getitem_208, add_51], dim=2)
        getitem_208 = add_51 = None
        k_25 = cat_27.type_as(v_12)
        cat_27 = None
        x_162 = torch._C._nn.scaled_dot_product_attention(
            q_25, k_25, v_12, attn_mask=None, dropout_p=0.0
        )
        q_25 = k_25 = v_12 = None
        transpose_13 = x_162.transpose(1, 2)
        x_162 = None
        x_163 = transpose_13.reshape(1, 197, 1024)
        transpose_13 = None
        x_164 = torch._C._nn.linear(
            x_163,
            l_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_,
        )
        x_163 = l_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_165 = torch.nn.functional.dropout(x_164, 0.0, False, False)
        x_164 = None
        mul_76 = l_self_modules_blocks_modules_12_parameters_gamma_1_ * x_165
        l_self_modules_blocks_modules_12_parameters_gamma_1_ = x_165 = None
        x_166 = x_160 + mul_76
        x_160 = mul_76 = None
        x_167 = torch.nn.functional.layer_norm(
            x_166,
            (1024,),
            l_self_modules_blocks_modules_12_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_12_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_12_modules_norm2_parameters_bias_
        ) = None
        x_168 = torch._C._nn.linear(
            x_167,
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_167 = (
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_169 = torch._C._nn.gelu(x_168, approximate="none")
        x_168 = None
        x_170 = torch.nn.functional.dropout(x_169, 0.0, False, False)
        x_169 = None
        x_171 = torch._C._nn.linear(
            x_170,
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_170 = (
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_172 = torch.nn.functional.dropout(x_171, 0.0, False, False)
        x_171 = None
        mul_77 = l_self_modules_blocks_modules_12_parameters_gamma_2_ * x_172
        l_self_modules_blocks_modules_12_parameters_gamma_2_ = x_172 = None
        x_173 = x_166 + mul_77
        x_166 = mul_77 = None
        getitem_214 = rot_pos_embed[13]
        x_174 = torch.nn.functional.layer_norm(
            x_173,
            (1024,),
            l_self_modules_blocks_modules_13_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_13_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_13_modules_norm1_parameters_bias_
        ) = None
        qkv_26 = torch._C._nn.linear(
            x_174,
            l_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_bias_,
        )
        x_174 = (
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
        getitem_218 = q_26[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_219 = q_26[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_26 = None
        tensor_split_26 = getitem_214.tensor_split(2, -1)
        sin_emb_27 = tensor_split_26[0]
        cos_emb_27 = tensor_split_26[1]
        tensor_split_26 = None
        mul_78 = getitem_219 * cos_emb_27
        cos_emb_27 = None
        getitem_222 = getitem_219[(Ellipsis, slice(1, None, 2))]
        neg_26 = -getitem_222
        getitem_222 = None
        getitem_223 = getitem_219[(Ellipsis, slice(None, None, 2))]
        getitem_219 = None
        stack_26 = torch.stack([neg_26, getitem_223], -1)
        neg_26 = getitem_223 = None
        reshape_53 = stack_26.reshape((1, 16, 196, 64))
        stack_26 = None
        mul_79 = reshape_53 * sin_emb_27
        reshape_53 = sin_emb_27 = None
        add_54 = mul_78 + mul_79
        mul_78 = mul_79 = None
        cat_28 = torch.cat([getitem_218, add_54], dim=2)
        getitem_218 = add_54 = None
        q_27 = cat_28.type_as(v_13)
        cat_28 = None
        getitem_224 = k_26[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_225 = k_26[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_26 = None
        tensor_split_27 = getitem_214.tensor_split(2, -1)
        getitem_214 = None
        sin_emb_28 = tensor_split_27[0]
        cos_emb_28 = tensor_split_27[1]
        tensor_split_27 = None
        mul_80 = getitem_225 * cos_emb_28
        cos_emb_28 = None
        getitem_228 = getitem_225[(Ellipsis, slice(1, None, 2))]
        neg_27 = -getitem_228
        getitem_228 = None
        getitem_229 = getitem_225[(Ellipsis, slice(None, None, 2))]
        getitem_225 = None
        stack_27 = torch.stack([neg_27, getitem_229], -1)
        neg_27 = getitem_229 = None
        reshape_54 = stack_27.reshape((1, 16, 196, 64))
        stack_27 = None
        mul_81 = reshape_54 * sin_emb_28
        reshape_54 = sin_emb_28 = None
        add_55 = mul_80 + mul_81
        mul_80 = mul_81 = None
        cat_29 = torch.cat([getitem_224, add_55], dim=2)
        getitem_224 = add_55 = None
        k_27 = cat_29.type_as(v_13)
        cat_29 = None
        x_175 = torch._C._nn.scaled_dot_product_attention(
            q_27, k_27, v_13, attn_mask=None, dropout_p=0.0
        )
        q_27 = k_27 = v_13 = None
        transpose_14 = x_175.transpose(1, 2)
        x_175 = None
        x_176 = transpose_14.reshape(1, 197, 1024)
        transpose_14 = None
        x_177 = torch._C._nn.linear(
            x_176,
            l_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_,
        )
        x_176 = l_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_178 = torch.nn.functional.dropout(x_177, 0.0, False, False)
        x_177 = None
        mul_82 = l_self_modules_blocks_modules_13_parameters_gamma_1_ * x_178
        l_self_modules_blocks_modules_13_parameters_gamma_1_ = x_178 = None
        x_179 = x_173 + mul_82
        x_173 = mul_82 = None
        x_180 = torch.nn.functional.layer_norm(
            x_179,
            (1024,),
            l_self_modules_blocks_modules_13_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_13_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_13_modules_norm2_parameters_bias_
        ) = None
        x_181 = torch._C._nn.linear(
            x_180,
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_180 = (
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_182 = torch._C._nn.gelu(x_181, approximate="none")
        x_181 = None
        x_183 = torch.nn.functional.dropout(x_182, 0.0, False, False)
        x_182 = None
        x_184 = torch._C._nn.linear(
            x_183,
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_183 = (
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_185 = torch.nn.functional.dropout(x_184, 0.0, False, False)
        x_184 = None
        mul_83 = l_self_modules_blocks_modules_13_parameters_gamma_2_ * x_185
        l_self_modules_blocks_modules_13_parameters_gamma_2_ = x_185 = None
        x_186 = x_179 + mul_83
        x_179 = mul_83 = None
        getitem_230 = rot_pos_embed[14]
        x_187 = torch.nn.functional.layer_norm(
            x_186,
            (1024,),
            l_self_modules_blocks_modules_14_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_14_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_14_modules_norm1_parameters_bias_
        ) = None
        qkv_28 = torch._C._nn.linear(
            x_187,
            l_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_bias_,
        )
        x_187 = (
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
        getitem_234 = q_28[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_235 = q_28[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_28 = None
        tensor_split_28 = getitem_230.tensor_split(2, -1)
        sin_emb_29 = tensor_split_28[0]
        cos_emb_29 = tensor_split_28[1]
        tensor_split_28 = None
        mul_84 = getitem_235 * cos_emb_29
        cos_emb_29 = None
        getitem_238 = getitem_235[(Ellipsis, slice(1, None, 2))]
        neg_28 = -getitem_238
        getitem_238 = None
        getitem_239 = getitem_235[(Ellipsis, slice(None, None, 2))]
        getitem_235 = None
        stack_28 = torch.stack([neg_28, getitem_239], -1)
        neg_28 = getitem_239 = None
        reshape_57 = stack_28.reshape((1, 16, 196, 64))
        stack_28 = None
        mul_85 = reshape_57 * sin_emb_29
        reshape_57 = sin_emb_29 = None
        add_58 = mul_84 + mul_85
        mul_84 = mul_85 = None
        cat_30 = torch.cat([getitem_234, add_58], dim=2)
        getitem_234 = add_58 = None
        q_29 = cat_30.type_as(v_14)
        cat_30 = None
        getitem_240 = k_28[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_241 = k_28[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_28 = None
        tensor_split_29 = getitem_230.tensor_split(2, -1)
        getitem_230 = None
        sin_emb_30 = tensor_split_29[0]
        cos_emb_30 = tensor_split_29[1]
        tensor_split_29 = None
        mul_86 = getitem_241 * cos_emb_30
        cos_emb_30 = None
        getitem_244 = getitem_241[(Ellipsis, slice(1, None, 2))]
        neg_29 = -getitem_244
        getitem_244 = None
        getitem_245 = getitem_241[(Ellipsis, slice(None, None, 2))]
        getitem_241 = None
        stack_29 = torch.stack([neg_29, getitem_245], -1)
        neg_29 = getitem_245 = None
        reshape_58 = stack_29.reshape((1, 16, 196, 64))
        stack_29 = None
        mul_87 = reshape_58 * sin_emb_30
        reshape_58 = sin_emb_30 = None
        add_59 = mul_86 + mul_87
        mul_86 = mul_87 = None
        cat_31 = torch.cat([getitem_240, add_59], dim=2)
        getitem_240 = add_59 = None
        k_29 = cat_31.type_as(v_14)
        cat_31 = None
        x_188 = torch._C._nn.scaled_dot_product_attention(
            q_29, k_29, v_14, attn_mask=None, dropout_p=0.0
        )
        q_29 = k_29 = v_14 = None
        transpose_15 = x_188.transpose(1, 2)
        x_188 = None
        x_189 = transpose_15.reshape(1, 197, 1024)
        transpose_15 = None
        x_190 = torch._C._nn.linear(
            x_189,
            l_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_bias_,
        )
        x_189 = l_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_191 = torch.nn.functional.dropout(x_190, 0.0, False, False)
        x_190 = None
        mul_88 = l_self_modules_blocks_modules_14_parameters_gamma_1_ * x_191
        l_self_modules_blocks_modules_14_parameters_gamma_1_ = x_191 = None
        x_192 = x_186 + mul_88
        x_186 = mul_88 = None
        x_193 = torch.nn.functional.layer_norm(
            x_192,
            (1024,),
            l_self_modules_blocks_modules_14_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_14_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_14_modules_norm2_parameters_bias_
        ) = None
        x_194 = torch._C._nn.linear(
            x_193,
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_193 = (
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_195 = torch._C._nn.gelu(x_194, approximate="none")
        x_194 = None
        x_196 = torch.nn.functional.dropout(x_195, 0.0, False, False)
        x_195 = None
        x_197 = torch._C._nn.linear(
            x_196,
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_196 = (
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_198 = torch.nn.functional.dropout(x_197, 0.0, False, False)
        x_197 = None
        mul_89 = l_self_modules_blocks_modules_14_parameters_gamma_2_ * x_198
        l_self_modules_blocks_modules_14_parameters_gamma_2_ = x_198 = None
        x_199 = x_192 + mul_89
        x_192 = mul_89 = None
        getitem_246 = rot_pos_embed[15]
        x_200 = torch.nn.functional.layer_norm(
            x_199,
            (1024,),
            l_self_modules_blocks_modules_15_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_15_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_15_modules_norm1_parameters_bias_
        ) = None
        qkv_30 = torch._C._nn.linear(
            x_200,
            l_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_bias_,
        )
        x_200 = (
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
        getitem_250 = q_30[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_251 = q_30[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_30 = None
        tensor_split_30 = getitem_246.tensor_split(2, -1)
        sin_emb_31 = tensor_split_30[0]
        cos_emb_31 = tensor_split_30[1]
        tensor_split_30 = None
        mul_90 = getitem_251 * cos_emb_31
        cos_emb_31 = None
        getitem_254 = getitem_251[(Ellipsis, slice(1, None, 2))]
        neg_30 = -getitem_254
        getitem_254 = None
        getitem_255 = getitem_251[(Ellipsis, slice(None, None, 2))]
        getitem_251 = None
        stack_30 = torch.stack([neg_30, getitem_255], -1)
        neg_30 = getitem_255 = None
        reshape_61 = stack_30.reshape((1, 16, 196, 64))
        stack_30 = None
        mul_91 = reshape_61 * sin_emb_31
        reshape_61 = sin_emb_31 = None
        add_62 = mul_90 + mul_91
        mul_90 = mul_91 = None
        cat_32 = torch.cat([getitem_250, add_62], dim=2)
        getitem_250 = add_62 = None
        q_31 = cat_32.type_as(v_15)
        cat_32 = None
        getitem_256 = k_30[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_257 = k_30[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_30 = None
        tensor_split_31 = getitem_246.tensor_split(2, -1)
        getitem_246 = None
        sin_emb_32 = tensor_split_31[0]
        cos_emb_32 = tensor_split_31[1]
        tensor_split_31 = None
        mul_92 = getitem_257 * cos_emb_32
        cos_emb_32 = None
        getitem_260 = getitem_257[(Ellipsis, slice(1, None, 2))]
        neg_31 = -getitem_260
        getitem_260 = None
        getitem_261 = getitem_257[(Ellipsis, slice(None, None, 2))]
        getitem_257 = None
        stack_31 = torch.stack([neg_31, getitem_261], -1)
        neg_31 = getitem_261 = None
        reshape_62 = stack_31.reshape((1, 16, 196, 64))
        stack_31 = None
        mul_93 = reshape_62 * sin_emb_32
        reshape_62 = sin_emb_32 = None
        add_63 = mul_92 + mul_93
        mul_92 = mul_93 = None
        cat_33 = torch.cat([getitem_256, add_63], dim=2)
        getitem_256 = add_63 = None
        k_31 = cat_33.type_as(v_15)
        cat_33 = None
        x_201 = torch._C._nn.scaled_dot_product_attention(
            q_31, k_31, v_15, attn_mask=None, dropout_p=0.0
        )
        q_31 = k_31 = v_15 = None
        transpose_16 = x_201.transpose(1, 2)
        x_201 = None
        x_202 = transpose_16.reshape(1, 197, 1024)
        transpose_16 = None
        x_203 = torch._C._nn.linear(
            x_202,
            l_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_bias_,
        )
        x_202 = l_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_204 = torch.nn.functional.dropout(x_203, 0.0, False, False)
        x_203 = None
        mul_94 = l_self_modules_blocks_modules_15_parameters_gamma_1_ * x_204
        l_self_modules_blocks_modules_15_parameters_gamma_1_ = x_204 = None
        x_205 = x_199 + mul_94
        x_199 = mul_94 = None
        x_206 = torch.nn.functional.layer_norm(
            x_205,
            (1024,),
            l_self_modules_blocks_modules_15_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_15_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_15_modules_norm2_parameters_bias_
        ) = None
        x_207 = torch._C._nn.linear(
            x_206,
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_206 = (
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_208 = torch._C._nn.gelu(x_207, approximate="none")
        x_207 = None
        x_209 = torch.nn.functional.dropout(x_208, 0.0, False, False)
        x_208 = None
        x_210 = torch._C._nn.linear(
            x_209,
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_209 = (
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_211 = torch.nn.functional.dropout(x_210, 0.0, False, False)
        x_210 = None
        mul_95 = l_self_modules_blocks_modules_15_parameters_gamma_2_ * x_211
        l_self_modules_blocks_modules_15_parameters_gamma_2_ = x_211 = None
        x_212 = x_205 + mul_95
        x_205 = mul_95 = None
        getitem_262 = rot_pos_embed[16]
        x_213 = torch.nn.functional.layer_norm(
            x_212,
            (1024,),
            l_self_modules_blocks_modules_16_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_16_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_16_modules_norm1_parameters_bias_
        ) = None
        qkv_32 = torch._C._nn.linear(
            x_213,
            l_self_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_bias_,
        )
        x_213 = (
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
        getitem_266 = q_32[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_267 = q_32[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_32 = None
        tensor_split_32 = getitem_262.tensor_split(2, -1)
        sin_emb_33 = tensor_split_32[0]
        cos_emb_33 = tensor_split_32[1]
        tensor_split_32 = None
        mul_96 = getitem_267 * cos_emb_33
        cos_emb_33 = None
        getitem_270 = getitem_267[(Ellipsis, slice(1, None, 2))]
        neg_32 = -getitem_270
        getitem_270 = None
        getitem_271 = getitem_267[(Ellipsis, slice(None, None, 2))]
        getitem_267 = None
        stack_32 = torch.stack([neg_32, getitem_271], -1)
        neg_32 = getitem_271 = None
        reshape_65 = stack_32.reshape((1, 16, 196, 64))
        stack_32 = None
        mul_97 = reshape_65 * sin_emb_33
        reshape_65 = sin_emb_33 = None
        add_66 = mul_96 + mul_97
        mul_96 = mul_97 = None
        cat_34 = torch.cat([getitem_266, add_66], dim=2)
        getitem_266 = add_66 = None
        q_33 = cat_34.type_as(v_16)
        cat_34 = None
        getitem_272 = k_32[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_273 = k_32[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_32 = None
        tensor_split_33 = getitem_262.tensor_split(2, -1)
        getitem_262 = None
        sin_emb_34 = tensor_split_33[0]
        cos_emb_34 = tensor_split_33[1]
        tensor_split_33 = None
        mul_98 = getitem_273 * cos_emb_34
        cos_emb_34 = None
        getitem_276 = getitem_273[(Ellipsis, slice(1, None, 2))]
        neg_33 = -getitem_276
        getitem_276 = None
        getitem_277 = getitem_273[(Ellipsis, slice(None, None, 2))]
        getitem_273 = None
        stack_33 = torch.stack([neg_33, getitem_277], -1)
        neg_33 = getitem_277 = None
        reshape_66 = stack_33.reshape((1, 16, 196, 64))
        stack_33 = None
        mul_99 = reshape_66 * sin_emb_34
        reshape_66 = sin_emb_34 = None
        add_67 = mul_98 + mul_99
        mul_98 = mul_99 = None
        cat_35 = torch.cat([getitem_272, add_67], dim=2)
        getitem_272 = add_67 = None
        k_33 = cat_35.type_as(v_16)
        cat_35 = None
        x_214 = torch._C._nn.scaled_dot_product_attention(
            q_33, k_33, v_16, attn_mask=None, dropout_p=0.0
        )
        q_33 = k_33 = v_16 = None
        transpose_17 = x_214.transpose(1, 2)
        x_214 = None
        x_215 = transpose_17.reshape(1, 197, 1024)
        transpose_17 = None
        x_216 = torch._C._nn.linear(
            x_215,
            l_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_bias_,
        )
        x_215 = l_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_217 = torch.nn.functional.dropout(x_216, 0.0, False, False)
        x_216 = None
        mul_100 = l_self_modules_blocks_modules_16_parameters_gamma_1_ * x_217
        l_self_modules_blocks_modules_16_parameters_gamma_1_ = x_217 = None
        x_218 = x_212 + mul_100
        x_212 = mul_100 = None
        x_219 = torch.nn.functional.layer_norm(
            x_218,
            (1024,),
            l_self_modules_blocks_modules_16_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_16_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_16_modules_norm2_parameters_bias_
        ) = None
        x_220 = torch._C._nn.linear(
            x_219,
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_219 = (
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_221 = torch._C._nn.gelu(x_220, approximate="none")
        x_220 = None
        x_222 = torch.nn.functional.dropout(x_221, 0.0, False, False)
        x_221 = None
        x_223 = torch._C._nn.linear(
            x_222,
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_222 = (
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_224 = torch.nn.functional.dropout(x_223, 0.0, False, False)
        x_223 = None
        mul_101 = l_self_modules_blocks_modules_16_parameters_gamma_2_ * x_224
        l_self_modules_blocks_modules_16_parameters_gamma_2_ = x_224 = None
        x_225 = x_218 + mul_101
        x_218 = mul_101 = None
        getitem_278 = rot_pos_embed[17]
        x_226 = torch.nn.functional.layer_norm(
            x_225,
            (1024,),
            l_self_modules_blocks_modules_17_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_17_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_17_modules_norm1_parameters_bias_
        ) = None
        qkv_34 = torch._C._nn.linear(
            x_226,
            l_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_bias_,
        )
        x_226 = (
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
        getitem_282 = q_34[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_283 = q_34[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_34 = None
        tensor_split_34 = getitem_278.tensor_split(2, -1)
        sin_emb_35 = tensor_split_34[0]
        cos_emb_35 = tensor_split_34[1]
        tensor_split_34 = None
        mul_102 = getitem_283 * cos_emb_35
        cos_emb_35 = None
        getitem_286 = getitem_283[(Ellipsis, slice(1, None, 2))]
        neg_34 = -getitem_286
        getitem_286 = None
        getitem_287 = getitem_283[(Ellipsis, slice(None, None, 2))]
        getitem_283 = None
        stack_34 = torch.stack([neg_34, getitem_287], -1)
        neg_34 = getitem_287 = None
        reshape_69 = stack_34.reshape((1, 16, 196, 64))
        stack_34 = None
        mul_103 = reshape_69 * sin_emb_35
        reshape_69 = sin_emb_35 = None
        add_70 = mul_102 + mul_103
        mul_102 = mul_103 = None
        cat_36 = torch.cat([getitem_282, add_70], dim=2)
        getitem_282 = add_70 = None
        q_35 = cat_36.type_as(v_17)
        cat_36 = None
        getitem_288 = k_34[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_289 = k_34[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_34 = None
        tensor_split_35 = getitem_278.tensor_split(2, -1)
        getitem_278 = None
        sin_emb_36 = tensor_split_35[0]
        cos_emb_36 = tensor_split_35[1]
        tensor_split_35 = None
        mul_104 = getitem_289 * cos_emb_36
        cos_emb_36 = None
        getitem_292 = getitem_289[(Ellipsis, slice(1, None, 2))]
        neg_35 = -getitem_292
        getitem_292 = None
        getitem_293 = getitem_289[(Ellipsis, slice(None, None, 2))]
        getitem_289 = None
        stack_35 = torch.stack([neg_35, getitem_293], -1)
        neg_35 = getitem_293 = None
        reshape_70 = stack_35.reshape((1, 16, 196, 64))
        stack_35 = None
        mul_105 = reshape_70 * sin_emb_36
        reshape_70 = sin_emb_36 = None
        add_71 = mul_104 + mul_105
        mul_104 = mul_105 = None
        cat_37 = torch.cat([getitem_288, add_71], dim=2)
        getitem_288 = add_71 = None
        k_35 = cat_37.type_as(v_17)
        cat_37 = None
        x_227 = torch._C._nn.scaled_dot_product_attention(
            q_35, k_35, v_17, attn_mask=None, dropout_p=0.0
        )
        q_35 = k_35 = v_17 = None
        transpose_18 = x_227.transpose(1, 2)
        x_227 = None
        x_228 = transpose_18.reshape(1, 197, 1024)
        transpose_18 = None
        x_229 = torch._C._nn.linear(
            x_228,
            l_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_bias_,
        )
        x_228 = l_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_230 = torch.nn.functional.dropout(x_229, 0.0, False, False)
        x_229 = None
        mul_106 = l_self_modules_blocks_modules_17_parameters_gamma_1_ * x_230
        l_self_modules_blocks_modules_17_parameters_gamma_1_ = x_230 = None
        x_231 = x_225 + mul_106
        x_225 = mul_106 = None
        x_232 = torch.nn.functional.layer_norm(
            x_231,
            (1024,),
            l_self_modules_blocks_modules_17_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_17_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_17_modules_norm2_parameters_bias_
        ) = None
        x_233 = torch._C._nn.linear(
            x_232,
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_232 = (
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_234 = torch._C._nn.gelu(x_233, approximate="none")
        x_233 = None
        x_235 = torch.nn.functional.dropout(x_234, 0.0, False, False)
        x_234 = None
        x_236 = torch._C._nn.linear(
            x_235,
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_235 = (
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_237 = torch.nn.functional.dropout(x_236, 0.0, False, False)
        x_236 = None
        mul_107 = l_self_modules_blocks_modules_17_parameters_gamma_2_ * x_237
        l_self_modules_blocks_modules_17_parameters_gamma_2_ = x_237 = None
        x_238 = x_231 + mul_107
        x_231 = mul_107 = None
        getitem_294 = rot_pos_embed[18]
        x_239 = torch.nn.functional.layer_norm(
            x_238,
            (1024,),
            l_self_modules_blocks_modules_18_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_18_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_18_modules_norm1_parameters_bias_
        ) = None
        qkv_36 = torch._C._nn.linear(
            x_239,
            l_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_bias_,
        )
        x_239 = (
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
        getitem_298 = q_36[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_299 = q_36[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_36 = None
        tensor_split_36 = getitem_294.tensor_split(2, -1)
        sin_emb_37 = tensor_split_36[0]
        cos_emb_37 = tensor_split_36[1]
        tensor_split_36 = None
        mul_108 = getitem_299 * cos_emb_37
        cos_emb_37 = None
        getitem_302 = getitem_299[(Ellipsis, slice(1, None, 2))]
        neg_36 = -getitem_302
        getitem_302 = None
        getitem_303 = getitem_299[(Ellipsis, slice(None, None, 2))]
        getitem_299 = None
        stack_36 = torch.stack([neg_36, getitem_303], -1)
        neg_36 = getitem_303 = None
        reshape_73 = stack_36.reshape((1, 16, 196, 64))
        stack_36 = None
        mul_109 = reshape_73 * sin_emb_37
        reshape_73 = sin_emb_37 = None
        add_74 = mul_108 + mul_109
        mul_108 = mul_109 = None
        cat_38 = torch.cat([getitem_298, add_74], dim=2)
        getitem_298 = add_74 = None
        q_37 = cat_38.type_as(v_18)
        cat_38 = None
        getitem_304 = k_36[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_305 = k_36[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_36 = None
        tensor_split_37 = getitem_294.tensor_split(2, -1)
        getitem_294 = None
        sin_emb_38 = tensor_split_37[0]
        cos_emb_38 = tensor_split_37[1]
        tensor_split_37 = None
        mul_110 = getitem_305 * cos_emb_38
        cos_emb_38 = None
        getitem_308 = getitem_305[(Ellipsis, slice(1, None, 2))]
        neg_37 = -getitem_308
        getitem_308 = None
        getitem_309 = getitem_305[(Ellipsis, slice(None, None, 2))]
        getitem_305 = None
        stack_37 = torch.stack([neg_37, getitem_309], -1)
        neg_37 = getitem_309 = None
        reshape_74 = stack_37.reshape((1, 16, 196, 64))
        stack_37 = None
        mul_111 = reshape_74 * sin_emb_38
        reshape_74 = sin_emb_38 = None
        add_75 = mul_110 + mul_111
        mul_110 = mul_111 = None
        cat_39 = torch.cat([getitem_304, add_75], dim=2)
        getitem_304 = add_75 = None
        k_37 = cat_39.type_as(v_18)
        cat_39 = None
        x_240 = torch._C._nn.scaled_dot_product_attention(
            q_37, k_37, v_18, attn_mask=None, dropout_p=0.0
        )
        q_37 = k_37 = v_18 = None
        transpose_19 = x_240.transpose(1, 2)
        x_240 = None
        x_241 = transpose_19.reshape(1, 197, 1024)
        transpose_19 = None
        x_242 = torch._C._nn.linear(
            x_241,
            l_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_bias_,
        )
        x_241 = l_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_243 = torch.nn.functional.dropout(x_242, 0.0, False, False)
        x_242 = None
        mul_112 = l_self_modules_blocks_modules_18_parameters_gamma_1_ * x_243
        l_self_modules_blocks_modules_18_parameters_gamma_1_ = x_243 = None
        x_244 = x_238 + mul_112
        x_238 = mul_112 = None
        x_245 = torch.nn.functional.layer_norm(
            x_244,
            (1024,),
            l_self_modules_blocks_modules_18_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_18_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_18_modules_norm2_parameters_bias_
        ) = None
        x_246 = torch._C._nn.linear(
            x_245,
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_245 = (
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_247 = torch._C._nn.gelu(x_246, approximate="none")
        x_246 = None
        x_248 = torch.nn.functional.dropout(x_247, 0.0, False, False)
        x_247 = None
        x_249 = torch._C._nn.linear(
            x_248,
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_248 = (
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_250 = torch.nn.functional.dropout(x_249, 0.0, False, False)
        x_249 = None
        mul_113 = l_self_modules_blocks_modules_18_parameters_gamma_2_ * x_250
        l_self_modules_blocks_modules_18_parameters_gamma_2_ = x_250 = None
        x_251 = x_244 + mul_113
        x_244 = mul_113 = None
        getitem_310 = rot_pos_embed[19]
        x_252 = torch.nn.functional.layer_norm(
            x_251,
            (1024,),
            l_self_modules_blocks_modules_19_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_19_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_19_modules_norm1_parameters_bias_
        ) = None
        qkv_38 = torch._C._nn.linear(
            x_252,
            l_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_bias_,
        )
        x_252 = (
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
        getitem_314 = q_38[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_315 = q_38[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_38 = None
        tensor_split_38 = getitem_310.tensor_split(2, -1)
        sin_emb_39 = tensor_split_38[0]
        cos_emb_39 = tensor_split_38[1]
        tensor_split_38 = None
        mul_114 = getitem_315 * cos_emb_39
        cos_emb_39 = None
        getitem_318 = getitem_315[(Ellipsis, slice(1, None, 2))]
        neg_38 = -getitem_318
        getitem_318 = None
        getitem_319 = getitem_315[(Ellipsis, slice(None, None, 2))]
        getitem_315 = None
        stack_38 = torch.stack([neg_38, getitem_319], -1)
        neg_38 = getitem_319 = None
        reshape_77 = stack_38.reshape((1, 16, 196, 64))
        stack_38 = None
        mul_115 = reshape_77 * sin_emb_39
        reshape_77 = sin_emb_39 = None
        add_78 = mul_114 + mul_115
        mul_114 = mul_115 = None
        cat_40 = torch.cat([getitem_314, add_78], dim=2)
        getitem_314 = add_78 = None
        q_39 = cat_40.type_as(v_19)
        cat_40 = None
        getitem_320 = k_38[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_321 = k_38[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_38 = None
        tensor_split_39 = getitem_310.tensor_split(2, -1)
        getitem_310 = None
        sin_emb_40 = tensor_split_39[0]
        cos_emb_40 = tensor_split_39[1]
        tensor_split_39 = None
        mul_116 = getitem_321 * cos_emb_40
        cos_emb_40 = None
        getitem_324 = getitem_321[(Ellipsis, slice(1, None, 2))]
        neg_39 = -getitem_324
        getitem_324 = None
        getitem_325 = getitem_321[(Ellipsis, slice(None, None, 2))]
        getitem_321 = None
        stack_39 = torch.stack([neg_39, getitem_325], -1)
        neg_39 = getitem_325 = None
        reshape_78 = stack_39.reshape((1, 16, 196, 64))
        stack_39 = None
        mul_117 = reshape_78 * sin_emb_40
        reshape_78 = sin_emb_40 = None
        add_79 = mul_116 + mul_117
        mul_116 = mul_117 = None
        cat_41 = torch.cat([getitem_320, add_79], dim=2)
        getitem_320 = add_79 = None
        k_39 = cat_41.type_as(v_19)
        cat_41 = None
        x_253 = torch._C._nn.scaled_dot_product_attention(
            q_39, k_39, v_19, attn_mask=None, dropout_p=0.0
        )
        q_39 = k_39 = v_19 = None
        transpose_20 = x_253.transpose(1, 2)
        x_253 = None
        x_254 = transpose_20.reshape(1, 197, 1024)
        transpose_20 = None
        x_255 = torch._C._nn.linear(
            x_254,
            l_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_bias_,
        )
        x_254 = l_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_256 = torch.nn.functional.dropout(x_255, 0.0, False, False)
        x_255 = None
        mul_118 = l_self_modules_blocks_modules_19_parameters_gamma_1_ * x_256
        l_self_modules_blocks_modules_19_parameters_gamma_1_ = x_256 = None
        x_257 = x_251 + mul_118
        x_251 = mul_118 = None
        x_258 = torch.nn.functional.layer_norm(
            x_257,
            (1024,),
            l_self_modules_blocks_modules_19_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_19_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_19_modules_norm2_parameters_bias_
        ) = None
        x_259 = torch._C._nn.linear(
            x_258,
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_258 = (
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_260 = torch._C._nn.gelu(x_259, approximate="none")
        x_259 = None
        x_261 = torch.nn.functional.dropout(x_260, 0.0, False, False)
        x_260 = None
        x_262 = torch._C._nn.linear(
            x_261,
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_261 = (
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_263 = torch.nn.functional.dropout(x_262, 0.0, False, False)
        x_262 = None
        mul_119 = l_self_modules_blocks_modules_19_parameters_gamma_2_ * x_263
        l_self_modules_blocks_modules_19_parameters_gamma_2_ = x_263 = None
        x_264 = x_257 + mul_119
        x_257 = mul_119 = None
        getitem_326 = rot_pos_embed[20]
        x_265 = torch.nn.functional.layer_norm(
            x_264,
            (1024,),
            l_self_modules_blocks_modules_20_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_20_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_20_modules_norm1_parameters_bias_
        ) = None
        qkv_40 = torch._C._nn.linear(
            x_265,
            l_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_bias_,
        )
        x_265 = (
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
        getitem_330 = q_40[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_331 = q_40[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_40 = None
        tensor_split_40 = getitem_326.tensor_split(2, -1)
        sin_emb_41 = tensor_split_40[0]
        cos_emb_41 = tensor_split_40[1]
        tensor_split_40 = None
        mul_120 = getitem_331 * cos_emb_41
        cos_emb_41 = None
        getitem_334 = getitem_331[(Ellipsis, slice(1, None, 2))]
        neg_40 = -getitem_334
        getitem_334 = None
        getitem_335 = getitem_331[(Ellipsis, slice(None, None, 2))]
        getitem_331 = None
        stack_40 = torch.stack([neg_40, getitem_335], -1)
        neg_40 = getitem_335 = None
        reshape_81 = stack_40.reshape((1, 16, 196, 64))
        stack_40 = None
        mul_121 = reshape_81 * sin_emb_41
        reshape_81 = sin_emb_41 = None
        add_82 = mul_120 + mul_121
        mul_120 = mul_121 = None
        cat_42 = torch.cat([getitem_330, add_82], dim=2)
        getitem_330 = add_82 = None
        q_41 = cat_42.type_as(v_20)
        cat_42 = None
        getitem_336 = k_40[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_337 = k_40[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_40 = None
        tensor_split_41 = getitem_326.tensor_split(2, -1)
        getitem_326 = None
        sin_emb_42 = tensor_split_41[0]
        cos_emb_42 = tensor_split_41[1]
        tensor_split_41 = None
        mul_122 = getitem_337 * cos_emb_42
        cos_emb_42 = None
        getitem_340 = getitem_337[(Ellipsis, slice(1, None, 2))]
        neg_41 = -getitem_340
        getitem_340 = None
        getitem_341 = getitem_337[(Ellipsis, slice(None, None, 2))]
        getitem_337 = None
        stack_41 = torch.stack([neg_41, getitem_341], -1)
        neg_41 = getitem_341 = None
        reshape_82 = stack_41.reshape((1, 16, 196, 64))
        stack_41 = None
        mul_123 = reshape_82 * sin_emb_42
        reshape_82 = sin_emb_42 = None
        add_83 = mul_122 + mul_123
        mul_122 = mul_123 = None
        cat_43 = torch.cat([getitem_336, add_83], dim=2)
        getitem_336 = add_83 = None
        k_41 = cat_43.type_as(v_20)
        cat_43 = None
        x_266 = torch._C._nn.scaled_dot_product_attention(
            q_41, k_41, v_20, attn_mask=None, dropout_p=0.0
        )
        q_41 = k_41 = v_20 = None
        transpose_21 = x_266.transpose(1, 2)
        x_266 = None
        x_267 = transpose_21.reshape(1, 197, 1024)
        transpose_21 = None
        x_268 = torch._C._nn.linear(
            x_267,
            l_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_bias_,
        )
        x_267 = l_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_269 = torch.nn.functional.dropout(x_268, 0.0, False, False)
        x_268 = None
        mul_124 = l_self_modules_blocks_modules_20_parameters_gamma_1_ * x_269
        l_self_modules_blocks_modules_20_parameters_gamma_1_ = x_269 = None
        x_270 = x_264 + mul_124
        x_264 = mul_124 = None
        x_271 = torch.nn.functional.layer_norm(
            x_270,
            (1024,),
            l_self_modules_blocks_modules_20_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_20_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_20_modules_norm2_parameters_bias_
        ) = None
        x_272 = torch._C._nn.linear(
            x_271,
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_271 = (
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_273 = torch._C._nn.gelu(x_272, approximate="none")
        x_272 = None
        x_274 = torch.nn.functional.dropout(x_273, 0.0, False, False)
        x_273 = None
        x_275 = torch._C._nn.linear(
            x_274,
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_274 = (
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_276 = torch.nn.functional.dropout(x_275, 0.0, False, False)
        x_275 = None
        mul_125 = l_self_modules_blocks_modules_20_parameters_gamma_2_ * x_276
        l_self_modules_blocks_modules_20_parameters_gamma_2_ = x_276 = None
        x_277 = x_270 + mul_125
        x_270 = mul_125 = None
        getitem_342 = rot_pos_embed[21]
        x_278 = torch.nn.functional.layer_norm(
            x_277,
            (1024,),
            l_self_modules_blocks_modules_21_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_21_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_21_modules_norm1_parameters_bias_
        ) = None
        qkv_42 = torch._C._nn.linear(
            x_278,
            l_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_bias_,
        )
        x_278 = (
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
        getitem_346 = q_42[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_347 = q_42[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_42 = None
        tensor_split_42 = getitem_342.tensor_split(2, -1)
        sin_emb_43 = tensor_split_42[0]
        cos_emb_43 = tensor_split_42[1]
        tensor_split_42 = None
        mul_126 = getitem_347 * cos_emb_43
        cos_emb_43 = None
        getitem_350 = getitem_347[(Ellipsis, slice(1, None, 2))]
        neg_42 = -getitem_350
        getitem_350 = None
        getitem_351 = getitem_347[(Ellipsis, slice(None, None, 2))]
        getitem_347 = None
        stack_42 = torch.stack([neg_42, getitem_351], -1)
        neg_42 = getitem_351 = None
        reshape_85 = stack_42.reshape((1, 16, 196, 64))
        stack_42 = None
        mul_127 = reshape_85 * sin_emb_43
        reshape_85 = sin_emb_43 = None
        add_86 = mul_126 + mul_127
        mul_126 = mul_127 = None
        cat_44 = torch.cat([getitem_346, add_86], dim=2)
        getitem_346 = add_86 = None
        q_43 = cat_44.type_as(v_21)
        cat_44 = None
        getitem_352 = k_42[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_353 = k_42[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_42 = None
        tensor_split_43 = getitem_342.tensor_split(2, -1)
        getitem_342 = None
        sin_emb_44 = tensor_split_43[0]
        cos_emb_44 = tensor_split_43[1]
        tensor_split_43 = None
        mul_128 = getitem_353 * cos_emb_44
        cos_emb_44 = None
        getitem_356 = getitem_353[(Ellipsis, slice(1, None, 2))]
        neg_43 = -getitem_356
        getitem_356 = None
        getitem_357 = getitem_353[(Ellipsis, slice(None, None, 2))]
        getitem_353 = None
        stack_43 = torch.stack([neg_43, getitem_357], -1)
        neg_43 = getitem_357 = None
        reshape_86 = stack_43.reshape((1, 16, 196, 64))
        stack_43 = None
        mul_129 = reshape_86 * sin_emb_44
        reshape_86 = sin_emb_44 = None
        add_87 = mul_128 + mul_129
        mul_128 = mul_129 = None
        cat_45 = torch.cat([getitem_352, add_87], dim=2)
        getitem_352 = add_87 = None
        k_43 = cat_45.type_as(v_21)
        cat_45 = None
        x_279 = torch._C._nn.scaled_dot_product_attention(
            q_43, k_43, v_21, attn_mask=None, dropout_p=0.0
        )
        q_43 = k_43 = v_21 = None
        transpose_22 = x_279.transpose(1, 2)
        x_279 = None
        x_280 = transpose_22.reshape(1, 197, 1024)
        transpose_22 = None
        x_281 = torch._C._nn.linear(
            x_280,
            l_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_bias_,
        )
        x_280 = l_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_282 = torch.nn.functional.dropout(x_281, 0.0, False, False)
        x_281 = None
        mul_130 = l_self_modules_blocks_modules_21_parameters_gamma_1_ * x_282
        l_self_modules_blocks_modules_21_parameters_gamma_1_ = x_282 = None
        x_283 = x_277 + mul_130
        x_277 = mul_130 = None
        x_284 = torch.nn.functional.layer_norm(
            x_283,
            (1024,),
            l_self_modules_blocks_modules_21_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_21_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_21_modules_norm2_parameters_bias_
        ) = None
        x_285 = torch._C._nn.linear(
            x_284,
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_284 = (
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_286 = torch._C._nn.gelu(x_285, approximate="none")
        x_285 = None
        x_287 = torch.nn.functional.dropout(x_286, 0.0, False, False)
        x_286 = None
        x_288 = torch._C._nn.linear(
            x_287,
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_287 = (
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_289 = torch.nn.functional.dropout(x_288, 0.0, False, False)
        x_288 = None
        mul_131 = l_self_modules_blocks_modules_21_parameters_gamma_2_ * x_289
        l_self_modules_blocks_modules_21_parameters_gamma_2_ = x_289 = None
        x_290 = x_283 + mul_131
        x_283 = mul_131 = None
        getitem_358 = rot_pos_embed[22]
        x_291 = torch.nn.functional.layer_norm(
            x_290,
            (1024,),
            l_self_modules_blocks_modules_22_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_22_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_22_modules_norm1_parameters_bias_
        ) = None
        qkv_44 = torch._C._nn.linear(
            x_291,
            l_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_bias_,
        )
        x_291 = (
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
        getitem_362 = q_44[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_363 = q_44[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_44 = None
        tensor_split_44 = getitem_358.tensor_split(2, -1)
        sin_emb_45 = tensor_split_44[0]
        cos_emb_45 = tensor_split_44[1]
        tensor_split_44 = None
        mul_132 = getitem_363 * cos_emb_45
        cos_emb_45 = None
        getitem_366 = getitem_363[(Ellipsis, slice(1, None, 2))]
        neg_44 = -getitem_366
        getitem_366 = None
        getitem_367 = getitem_363[(Ellipsis, slice(None, None, 2))]
        getitem_363 = None
        stack_44 = torch.stack([neg_44, getitem_367], -1)
        neg_44 = getitem_367 = None
        reshape_89 = stack_44.reshape((1, 16, 196, 64))
        stack_44 = None
        mul_133 = reshape_89 * sin_emb_45
        reshape_89 = sin_emb_45 = None
        add_90 = mul_132 + mul_133
        mul_132 = mul_133 = None
        cat_46 = torch.cat([getitem_362, add_90], dim=2)
        getitem_362 = add_90 = None
        q_45 = cat_46.type_as(v_22)
        cat_46 = None
        getitem_368 = k_44[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_369 = k_44[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_44 = None
        tensor_split_45 = getitem_358.tensor_split(2, -1)
        getitem_358 = None
        sin_emb_46 = tensor_split_45[0]
        cos_emb_46 = tensor_split_45[1]
        tensor_split_45 = None
        mul_134 = getitem_369 * cos_emb_46
        cos_emb_46 = None
        getitem_372 = getitem_369[(Ellipsis, slice(1, None, 2))]
        neg_45 = -getitem_372
        getitem_372 = None
        getitem_373 = getitem_369[(Ellipsis, slice(None, None, 2))]
        getitem_369 = None
        stack_45 = torch.stack([neg_45, getitem_373], -1)
        neg_45 = getitem_373 = None
        reshape_90 = stack_45.reshape((1, 16, 196, 64))
        stack_45 = None
        mul_135 = reshape_90 * sin_emb_46
        reshape_90 = sin_emb_46 = None
        add_91 = mul_134 + mul_135
        mul_134 = mul_135 = None
        cat_47 = torch.cat([getitem_368, add_91], dim=2)
        getitem_368 = add_91 = None
        k_45 = cat_47.type_as(v_22)
        cat_47 = None
        x_292 = torch._C._nn.scaled_dot_product_attention(
            q_45, k_45, v_22, attn_mask=None, dropout_p=0.0
        )
        q_45 = k_45 = v_22 = None
        transpose_23 = x_292.transpose(1, 2)
        x_292 = None
        x_293 = transpose_23.reshape(1, 197, 1024)
        transpose_23 = None
        x_294 = torch._C._nn.linear(
            x_293,
            l_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_bias_,
        )
        x_293 = l_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_295 = torch.nn.functional.dropout(x_294, 0.0, False, False)
        x_294 = None
        mul_136 = l_self_modules_blocks_modules_22_parameters_gamma_1_ * x_295
        l_self_modules_blocks_modules_22_parameters_gamma_1_ = x_295 = None
        x_296 = x_290 + mul_136
        x_290 = mul_136 = None
        x_297 = torch.nn.functional.layer_norm(
            x_296,
            (1024,),
            l_self_modules_blocks_modules_22_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_22_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_22_modules_norm2_parameters_bias_
        ) = None
        x_298 = torch._C._nn.linear(
            x_297,
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_297 = (
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_299 = torch._C._nn.gelu(x_298, approximate="none")
        x_298 = None
        x_300 = torch.nn.functional.dropout(x_299, 0.0, False, False)
        x_299 = None
        x_301 = torch._C._nn.linear(
            x_300,
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_300 = (
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_302 = torch.nn.functional.dropout(x_301, 0.0, False, False)
        x_301 = None
        mul_137 = l_self_modules_blocks_modules_22_parameters_gamma_2_ * x_302
        l_self_modules_blocks_modules_22_parameters_gamma_2_ = x_302 = None
        x_303 = x_296 + mul_137
        x_296 = mul_137 = None
        getitem_374 = rot_pos_embed[23]
        rot_pos_embed = None
        x_304 = torch.nn.functional.layer_norm(
            x_303,
            (1024,),
            l_self_modules_blocks_modules_23_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_23_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_23_modules_norm1_parameters_bias_
        ) = None
        qkv_46 = torch._C._nn.linear(
            x_304,
            l_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_bias_,
        )
        x_304 = (
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
        getitem_378 = q_46[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_379 = q_46[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_46 = None
        tensor_split_46 = getitem_374.tensor_split(2, -1)
        sin_emb_47 = tensor_split_46[0]
        cos_emb_47 = tensor_split_46[1]
        tensor_split_46 = None
        mul_138 = getitem_379 * cos_emb_47
        cos_emb_47 = None
        getitem_382 = getitem_379[(Ellipsis, slice(1, None, 2))]
        neg_46 = -getitem_382
        getitem_382 = None
        getitem_383 = getitem_379[(Ellipsis, slice(None, None, 2))]
        getitem_379 = None
        stack_46 = torch.stack([neg_46, getitem_383], -1)
        neg_46 = getitem_383 = None
        reshape_93 = stack_46.reshape((1, 16, 196, 64))
        stack_46 = None
        mul_139 = reshape_93 * sin_emb_47
        reshape_93 = sin_emb_47 = None
        add_94 = mul_138 + mul_139
        mul_138 = mul_139 = None
        cat_48 = torch.cat([getitem_378, add_94], dim=2)
        getitem_378 = add_94 = None
        q_47 = cat_48.type_as(v_23)
        cat_48 = None
        getitem_384 = k_46[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_385 = k_46[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_46 = None
        tensor_split_47 = getitem_374.tensor_split(2, -1)
        getitem_374 = None
        sin_emb_48 = tensor_split_47[0]
        cos_emb_48 = tensor_split_47[1]
        tensor_split_47 = None
        mul_140 = getitem_385 * cos_emb_48
        cos_emb_48 = None
        getitem_388 = getitem_385[(Ellipsis, slice(1, None, 2))]
        neg_47 = -getitem_388
        getitem_388 = None
        getitem_389 = getitem_385[(Ellipsis, slice(None, None, 2))]
        getitem_385 = None
        stack_47 = torch.stack([neg_47, getitem_389], -1)
        neg_47 = getitem_389 = None
        reshape_94 = stack_47.reshape((1, 16, 196, 64))
        stack_47 = None
        mul_141 = reshape_94 * sin_emb_48
        reshape_94 = sin_emb_48 = None
        add_95 = mul_140 + mul_141
        mul_140 = mul_141 = None
        cat_49 = torch.cat([getitem_384, add_95], dim=2)
        getitem_384 = add_95 = None
        k_47 = cat_49.type_as(v_23)
        cat_49 = None
        x_305 = torch._C._nn.scaled_dot_product_attention(
            q_47, k_47, v_23, attn_mask=None, dropout_p=0.0
        )
        q_47 = k_47 = v_23 = None
        transpose_24 = x_305.transpose(1, 2)
        x_305 = None
        x_306 = transpose_24.reshape(1, 197, 1024)
        transpose_24 = None
        x_307 = torch._C._nn.linear(
            x_306,
            l_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_bias_,
        )
        x_306 = l_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_308 = torch.nn.functional.dropout(x_307, 0.0, False, False)
        x_307 = None
        mul_142 = l_self_modules_blocks_modules_23_parameters_gamma_1_ * x_308
        l_self_modules_blocks_modules_23_parameters_gamma_1_ = x_308 = None
        x_309 = x_303 + mul_142
        x_303 = mul_142 = None
        x_310 = torch.nn.functional.layer_norm(
            x_309,
            (1024,),
            l_self_modules_blocks_modules_23_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_23_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_23_modules_norm2_parameters_bias_
        ) = None
        x_311 = torch._C._nn.linear(
            x_310,
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_310 = (
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_312 = torch._C._nn.gelu(x_311, approximate="none")
        x_311 = None
        x_313 = torch.nn.functional.dropout(x_312, 0.0, False, False)
        x_312 = None
        x_314 = torch._C._nn.linear(
            x_313,
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_313 = (
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_315 = torch.nn.functional.dropout(x_314, 0.0, False, False)
        x_314 = None
        mul_143 = l_self_modules_blocks_modules_23_parameters_gamma_2_ * x_315
        l_self_modules_blocks_modules_23_parameters_gamma_2_ = x_315 = None
        x_316 = x_309 + mul_143
        x_309 = mul_143 = None
        x_317 = torch.nn.functional.layer_norm(
            x_316,
            (1024,),
            l_self_modules_norm_parameters_weight_,
            l_self_modules_norm_parameters_bias_,
            1e-06,
        )
        x_316 = (
            l_self_modules_norm_parameters_weight_
        ) = l_self_modules_norm_parameters_bias_ = None
        x_318 = x_317[(slice(None, None, None), 0)]
        x_317 = None
        x_319 = torch.nn.functional.dropout(x_318, 0.0, False, False)
        x_318 = None
        x_320 = torch._C._nn.linear(
            x_319,
            l_self_modules_head_parameters_weight_,
            l_self_modules_head_parameters_bias_,
        )
        x_319 = (
            l_self_modules_head_parameters_weight_
        ) = l_self_modules_head_parameters_bias_ = None
        return (x_320,)
