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
        L_self_modules_blocks_modules_0_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_1_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_2_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_3_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_4_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_5_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_6_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_7_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_8_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_9_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_10_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_11_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_12_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_13_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_14_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_15_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_16_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_17_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_18_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_19_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_20_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_21_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_22_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_23_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_24_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_24_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_25_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_25_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_26_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_26_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_27_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_27_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_28_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_28_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_29_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_29_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_30_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_30_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_31_modules_attn_parameters_rel_pos_h_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_31_modules_attn_parameters_rel_pos_w_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_31_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_31_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_31_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_31_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_31_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_31_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_31_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_31_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_blocks_modules_0_modules_attn_parameters_rel_pos_h_ = (
            L_self_modules_blocks_modules_0_modules_attn_parameters_rel_pos_h_
        )
        l_self_modules_blocks_modules_0_modules_attn_parameters_rel_pos_w_ = (
            L_self_modules_blocks_modules_0_modules_attn_parameters_rel_pos_w_
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
        l_self_modules_blocks_modules_1_modules_attn_parameters_rel_pos_h_ = (
            L_self_modules_blocks_modules_1_modules_attn_parameters_rel_pos_h_
        )
        l_self_modules_blocks_modules_1_modules_attn_parameters_rel_pos_w_ = (
            L_self_modules_blocks_modules_1_modules_attn_parameters_rel_pos_w_
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
        l_self_modules_blocks_modules_2_modules_attn_parameters_rel_pos_h_ = (
            L_self_modules_blocks_modules_2_modules_attn_parameters_rel_pos_h_
        )
        l_self_modules_blocks_modules_2_modules_attn_parameters_rel_pos_w_ = (
            L_self_modules_blocks_modules_2_modules_attn_parameters_rel_pos_w_
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
        l_self_modules_blocks_modules_3_modules_attn_parameters_rel_pos_h_ = (
            L_self_modules_blocks_modules_3_modules_attn_parameters_rel_pos_h_
        )
        l_self_modules_blocks_modules_3_modules_attn_parameters_rel_pos_w_ = (
            L_self_modules_blocks_modules_3_modules_attn_parameters_rel_pos_w_
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
        l_self_modules_blocks_modules_4_modules_attn_parameters_rel_pos_h_ = (
            L_self_modules_blocks_modules_4_modules_attn_parameters_rel_pos_h_
        )
        l_self_modules_blocks_modules_4_modules_attn_parameters_rel_pos_w_ = (
            L_self_modules_blocks_modules_4_modules_attn_parameters_rel_pos_w_
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
        l_self_modules_blocks_modules_5_modules_attn_parameters_rel_pos_h_ = (
            L_self_modules_blocks_modules_5_modules_attn_parameters_rel_pos_h_
        )
        l_self_modules_blocks_modules_5_modules_attn_parameters_rel_pos_w_ = (
            L_self_modules_blocks_modules_5_modules_attn_parameters_rel_pos_w_
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
        l_self_modules_blocks_modules_6_modules_attn_parameters_rel_pos_h_ = (
            L_self_modules_blocks_modules_6_modules_attn_parameters_rel_pos_h_
        )
        l_self_modules_blocks_modules_6_modules_attn_parameters_rel_pos_w_ = (
            L_self_modules_blocks_modules_6_modules_attn_parameters_rel_pos_w_
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
        l_self_modules_blocks_modules_7_modules_attn_parameters_rel_pos_h_ = (
            L_self_modules_blocks_modules_7_modules_attn_parameters_rel_pos_h_
        )
        l_self_modules_blocks_modules_7_modules_attn_parameters_rel_pos_w_ = (
            L_self_modules_blocks_modules_7_modules_attn_parameters_rel_pos_w_
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
        l_self_modules_blocks_modules_8_modules_attn_parameters_rel_pos_h_ = (
            L_self_modules_blocks_modules_8_modules_attn_parameters_rel_pos_h_
        )
        l_self_modules_blocks_modules_8_modules_attn_parameters_rel_pos_w_ = (
            L_self_modules_blocks_modules_8_modules_attn_parameters_rel_pos_w_
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
        l_self_modules_blocks_modules_9_modules_attn_parameters_rel_pos_h_ = (
            L_self_modules_blocks_modules_9_modules_attn_parameters_rel_pos_h_
        )
        l_self_modules_blocks_modules_9_modules_attn_parameters_rel_pos_w_ = (
            L_self_modules_blocks_modules_9_modules_attn_parameters_rel_pos_w_
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
        l_self_modules_blocks_modules_10_modules_attn_parameters_rel_pos_h_ = (
            L_self_modules_blocks_modules_10_modules_attn_parameters_rel_pos_h_
        )
        l_self_modules_blocks_modules_10_modules_attn_parameters_rel_pos_w_ = (
            L_self_modules_blocks_modules_10_modules_attn_parameters_rel_pos_w_
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
        l_self_modules_blocks_modules_11_modules_attn_parameters_rel_pos_h_ = (
            L_self_modules_blocks_modules_11_modules_attn_parameters_rel_pos_h_
        )
        l_self_modules_blocks_modules_11_modules_attn_parameters_rel_pos_w_ = (
            L_self_modules_blocks_modules_11_modules_attn_parameters_rel_pos_w_
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
        l_self_modules_blocks_modules_12_modules_attn_parameters_rel_pos_h_ = (
            L_self_modules_blocks_modules_12_modules_attn_parameters_rel_pos_h_
        )
        l_self_modules_blocks_modules_12_modules_attn_parameters_rel_pos_w_ = (
            L_self_modules_blocks_modules_12_modules_attn_parameters_rel_pos_w_
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
        l_self_modules_blocks_modules_13_modules_attn_parameters_rel_pos_h_ = (
            L_self_modules_blocks_modules_13_modules_attn_parameters_rel_pos_h_
        )
        l_self_modules_blocks_modules_13_modules_attn_parameters_rel_pos_w_ = (
            L_self_modules_blocks_modules_13_modules_attn_parameters_rel_pos_w_
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
        l_self_modules_blocks_modules_14_modules_attn_parameters_rel_pos_h_ = (
            L_self_modules_blocks_modules_14_modules_attn_parameters_rel_pos_h_
        )
        l_self_modules_blocks_modules_14_modules_attn_parameters_rel_pos_w_ = (
            L_self_modules_blocks_modules_14_modules_attn_parameters_rel_pos_w_
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
        l_self_modules_blocks_modules_15_modules_attn_parameters_rel_pos_h_ = (
            L_self_modules_blocks_modules_15_modules_attn_parameters_rel_pos_h_
        )
        l_self_modules_blocks_modules_15_modules_attn_parameters_rel_pos_w_ = (
            L_self_modules_blocks_modules_15_modules_attn_parameters_rel_pos_w_
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
        l_self_modules_blocks_modules_16_modules_attn_parameters_rel_pos_h_ = (
            L_self_modules_blocks_modules_16_modules_attn_parameters_rel_pos_h_
        )
        l_self_modules_blocks_modules_16_modules_attn_parameters_rel_pos_w_ = (
            L_self_modules_blocks_modules_16_modules_attn_parameters_rel_pos_w_
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
        l_self_modules_blocks_modules_17_modules_attn_parameters_rel_pos_h_ = (
            L_self_modules_blocks_modules_17_modules_attn_parameters_rel_pos_h_
        )
        l_self_modules_blocks_modules_17_modules_attn_parameters_rel_pos_w_ = (
            L_self_modules_blocks_modules_17_modules_attn_parameters_rel_pos_w_
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
        l_self_modules_blocks_modules_18_modules_attn_parameters_rel_pos_h_ = (
            L_self_modules_blocks_modules_18_modules_attn_parameters_rel_pos_h_
        )
        l_self_modules_blocks_modules_18_modules_attn_parameters_rel_pos_w_ = (
            L_self_modules_blocks_modules_18_modules_attn_parameters_rel_pos_w_
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
        l_self_modules_blocks_modules_19_modules_attn_parameters_rel_pos_h_ = (
            L_self_modules_blocks_modules_19_modules_attn_parameters_rel_pos_h_
        )
        l_self_modules_blocks_modules_19_modules_attn_parameters_rel_pos_w_ = (
            L_self_modules_blocks_modules_19_modules_attn_parameters_rel_pos_w_
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
        l_self_modules_blocks_modules_20_modules_attn_parameters_rel_pos_h_ = (
            L_self_modules_blocks_modules_20_modules_attn_parameters_rel_pos_h_
        )
        l_self_modules_blocks_modules_20_modules_attn_parameters_rel_pos_w_ = (
            L_self_modules_blocks_modules_20_modules_attn_parameters_rel_pos_w_
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
        l_self_modules_blocks_modules_21_modules_attn_parameters_rel_pos_h_ = (
            L_self_modules_blocks_modules_21_modules_attn_parameters_rel_pos_h_
        )
        l_self_modules_blocks_modules_21_modules_attn_parameters_rel_pos_w_ = (
            L_self_modules_blocks_modules_21_modules_attn_parameters_rel_pos_w_
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
        l_self_modules_blocks_modules_22_modules_attn_parameters_rel_pos_h_ = (
            L_self_modules_blocks_modules_22_modules_attn_parameters_rel_pos_h_
        )
        l_self_modules_blocks_modules_22_modules_attn_parameters_rel_pos_w_ = (
            L_self_modules_blocks_modules_22_modules_attn_parameters_rel_pos_w_
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
        l_self_modules_blocks_modules_23_modules_attn_parameters_rel_pos_h_ = (
            L_self_modules_blocks_modules_23_modules_attn_parameters_rel_pos_h_
        )
        l_self_modules_blocks_modules_23_modules_attn_parameters_rel_pos_w_ = (
            L_self_modules_blocks_modules_23_modules_attn_parameters_rel_pos_w_
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
        l_self_modules_blocks_modules_24_modules_attn_parameters_rel_pos_h_ = (
            L_self_modules_blocks_modules_24_modules_attn_parameters_rel_pos_h_
        )
        l_self_modules_blocks_modules_24_modules_attn_parameters_rel_pos_w_ = (
            L_self_modules_blocks_modules_24_modules_attn_parameters_rel_pos_w_
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
        l_self_modules_blocks_modules_25_modules_attn_parameters_rel_pos_h_ = (
            L_self_modules_blocks_modules_25_modules_attn_parameters_rel_pos_h_
        )
        l_self_modules_blocks_modules_25_modules_attn_parameters_rel_pos_w_ = (
            L_self_modules_blocks_modules_25_modules_attn_parameters_rel_pos_w_
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
        l_self_modules_blocks_modules_26_modules_attn_parameters_rel_pos_h_ = (
            L_self_modules_blocks_modules_26_modules_attn_parameters_rel_pos_h_
        )
        l_self_modules_blocks_modules_26_modules_attn_parameters_rel_pos_w_ = (
            L_self_modules_blocks_modules_26_modules_attn_parameters_rel_pos_w_
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
        l_self_modules_blocks_modules_27_modules_attn_parameters_rel_pos_h_ = (
            L_self_modules_blocks_modules_27_modules_attn_parameters_rel_pos_h_
        )
        l_self_modules_blocks_modules_27_modules_attn_parameters_rel_pos_w_ = (
            L_self_modules_blocks_modules_27_modules_attn_parameters_rel_pos_w_
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
        l_self_modules_blocks_modules_28_modules_attn_parameters_rel_pos_h_ = (
            L_self_modules_blocks_modules_28_modules_attn_parameters_rel_pos_h_
        )
        l_self_modules_blocks_modules_28_modules_attn_parameters_rel_pos_w_ = (
            L_self_modules_blocks_modules_28_modules_attn_parameters_rel_pos_w_
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
        l_self_modules_blocks_modules_29_modules_attn_parameters_rel_pos_h_ = (
            L_self_modules_blocks_modules_29_modules_attn_parameters_rel_pos_h_
        )
        l_self_modules_blocks_modules_29_modules_attn_parameters_rel_pos_w_ = (
            L_self_modules_blocks_modules_29_modules_attn_parameters_rel_pos_w_
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
        l_self_modules_blocks_modules_30_modules_attn_parameters_rel_pos_h_ = (
            L_self_modules_blocks_modules_30_modules_attn_parameters_rel_pos_h_
        )
        l_self_modules_blocks_modules_30_modules_attn_parameters_rel_pos_w_ = (
            L_self_modules_blocks_modules_30_modules_attn_parameters_rel_pos_w_
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
        l_self_modules_blocks_modules_31_modules_attn_parameters_rel_pos_h_ = (
            L_self_modules_blocks_modules_31_modules_attn_parameters_rel_pos_h_
        )
        l_self_modules_blocks_modules_31_modules_attn_parameters_rel_pos_w_ = (
            L_self_modules_blocks_modules_31_modules_attn_parameters_rel_pos_w_
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
        l_self_modules_neck_modules_0_parameters_weight_ = (
            L_self_modules_neck_modules_0_parameters_weight_
        )
        l_self_modules_neck_modules_1_parameters_weight_ = (
            L_self_modules_neck_modules_1_parameters_weight_
        )
        l_self_modules_neck_modules_1_parameters_bias_ = (
            L_self_modules_neck_modules_1_parameters_bias_
        )
        l_self_modules_neck_modules_2_parameters_weight_ = (
            L_self_modules_neck_modules_2_parameters_weight_
        )
        l_self_modules_neck_modules_3_parameters_weight_ = (
            L_self_modules_neck_modules_3_parameters_weight_
        )
        l_self_modules_neck_modules_3_parameters_bias_ = (
            L_self_modules_neck_modules_3_parameters_bias_
        )
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
        x_1 = x.permute(0, 2, 3, 1)
        x = None
        posemb = l_self_parameters_pos_embed_.float()
        l_self_parameters_pos_embed_ = None
        reshape = posemb.reshape(1, 64, 64, 1280)
        posemb = None
        posemb_1 = reshape.permute(0, 3, 1, 2)
        reshape = None
        posemb_2 = torch.nn.functional.interpolate(
            posemb_1, size=(14, 14), mode="bicubic", antialias=True
        )
        posemb_1 = None
        permute_2 = posemb_2.permute(0, 2, 3, 1)
        posemb_2 = None
        posemb_3 = permute_2.to(torch.float32)
        permute_2 = None
        x_2 = x_1 + posemb_3
        x_1 = posemb_3 = None
        x_3 = torch.nn.functional.dropout(x_2, 0.0, False, False)
        x_2 = None
        x_4 = torch.nn.functional.layer_norm(
            x_3,
            (1280,),
            l_self_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_0_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_0_modules_norm1_parameters_bias_
        ) = None
        x_5 = torch._C._nn.pad(x_4, (0, 0, 0, 0, 0, 0), "constant", None)
        x_4 = None
        x_6 = x_5.view(1, 1, 14, 1, 14, 1280)
        x_5 = None
        permute_3 = x_6.permute(0, 1, 3, 2, 4, 5)
        x_6 = None
        contiguous = permute_3.contiguous()
        permute_3 = None
        windows = contiguous.view(-1, 14, 14, 1280)
        contiguous = None
        x_7 = windows.reshape(1, 196, -1)
        windows = None
        linear = torch._C._nn.linear(
            x_7,
            l_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_,
        )
        x_7 = (
            l_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_2 = linear.view(1, 196, 3, 16, -1)
        linear = None
        qkv = view_2.permute(2, 0, 3, 1, 4)
        view_2 = None
        reshape_2 = qkv.reshape(3, 16, 196, -1)
        qkv = None
        unbind = reshape_2.unbind(0)
        reshape_2 = None
        q = unbind[0]
        k = unbind[1]
        v = unbind[2]
        unbind = None
        arange = torch.arange(14)
        getitem_7 = arange[(slice(None, None, None), None)]
        arange = None
        q_coords = getitem_7 * 1.0
        getitem_7 = None
        arange_1 = torch.arange(14)
        getitem_8 = arange_1[(None, slice(None, None, None))]
        arange_1 = None
        k_coords = getitem_8 * 1.0
        getitem_8 = None
        sub = q_coords - k_coords
        q_coords = k_coords = None
        relative_coords = sub + 13.0
        sub = None
        long = relative_coords.long()
        relative_coords = None
        Rh = l_self_modules_blocks_modules_0_modules_attn_parameters_rel_pos_h_[long]
        l_self_modules_blocks_modules_0_modules_attn_parameters_rel_pos_h_ = long = None
        arange_2 = torch.arange(14)
        getitem_10 = arange_2[(slice(None, None, None), None)]
        arange_2 = None
        q_coords_1 = getitem_10 * 1.0
        getitem_10 = None
        arange_3 = torch.arange(14)
        getitem_11 = arange_3[(None, slice(None, None, None))]
        arange_3 = None
        k_coords_1 = getitem_11 * 1.0
        getitem_11 = None
        sub_1 = q_coords_1 - k_coords_1
        q_coords_1 = k_coords_1 = None
        relative_coords_1 = sub_1 + 13.0
        sub_1 = None
        long_1 = relative_coords_1.long()
        relative_coords_1 = None
        Rw = l_self_modules_blocks_modules_0_modules_attn_parameters_rel_pos_w_[long_1]
        l_self_modules_blocks_modules_0_modules_attn_parameters_rel_pos_w_ = (
            long_1
        ) = None
        r_q = q.reshape(16, 14, 14, 80)
        rel_h = torch.functional.einsum("bhwc,hkc->bhwk", r_q, Rh)
        Rh = None
        rel_w = torch.functional.einsum("bhwc,wkc->bhwk", r_q, Rw)
        r_q = Rw = None
        getitem_13 = rel_h[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h = None
        getitem_14 = rel_w[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w = None
        attn_bias = getitem_13 + getitem_14
        getitem_13 = getitem_14 = None
        attn_bias_1 = attn_bias.reshape(-1, 196, 196)
        attn_bias = None
        x_8 = torch._C._nn.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias_1, dropout_p=0.0
        )
        q = k = v = attn_bias_1 = None
        view_3 = x_8.view(1, 16, 196, -1)
        x_8 = None
        transpose = view_3.transpose(1, 2)
        view_3 = None
        x_9 = transpose.reshape(1, 196, -1)
        transpose = None
        x_10 = torch._C._nn.linear(
            x_9,
            l_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_9 = (
            l_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_11 = torch.nn.functional.dropout(x_10, 0.0, False, False)
        x_10 = None
        x_12 = x_11.view(1, 14, 14, -1)
        x_11 = None
        x_13 = x_12.view(1, 1, 1, 14, 14, -1)
        x_12 = None
        permute_5 = x_13.permute(0, 1, 3, 2, 4, 5)
        x_13 = None
        contiguous_1 = permute_5.contiguous()
        permute_5 = None
        x_14 = contiguous_1.view(1, 14, 14, -1)
        contiguous_1 = None
        getitem_15 = x_14[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_14 = None
        x_15 = getitem_15.contiguous()
        getitem_15 = None
        x_16 = x_3 + x_15
        x_3 = x_15 = None
        x_17 = x_16.reshape(1, 196, -1)
        x_16 = None
        layer_norm_1 = torch.nn.functional.layer_norm(
            x_17,
            (1280,),
            l_self_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_0_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_0_modules_norm2_parameters_bias_
        ) = None
        x_18 = torch._C._nn.linear(
            layer_norm_1,
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_1 = (
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_19 = torch._C._nn.gelu(x_18, approximate="none")
        x_18 = None
        x_20 = torch.nn.functional.dropout(x_19, 0.0, False, False)
        x_19 = None
        x_21 = torch._C._nn.linear(
            x_20,
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_20 = (
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_22 = torch.nn.functional.dropout(x_21, 0.0, False, False)
        x_21 = None
        x_23 = x_17 + x_22
        x_17 = x_22 = None
        x_24 = x_23.reshape(1, 14, 14, -1)
        x_23 = None
        x_25 = torch.nn.functional.layer_norm(
            x_24,
            (1280,),
            l_self_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_1_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_1_modules_norm1_parameters_bias_
        ) = None
        x_26 = torch._C._nn.pad(x_25, (0, 0, 0, 0, 0, 0), "constant", None)
        x_25 = None
        x_27 = x_26.view(1, 1, 14, 1, 14, 1280)
        x_26 = None
        permute_6 = x_27.permute(0, 1, 3, 2, 4, 5)
        x_27 = None
        contiguous_3 = permute_6.contiguous()
        permute_6 = None
        windows_1 = contiguous_3.view(-1, 14, 14, 1280)
        contiguous_3 = None
        x_28 = windows_1.reshape(1, 196, -1)
        windows_1 = None
        linear_4 = torch._C._nn.linear(
            x_28,
            l_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_,
        )
        x_28 = (
            l_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_9 = linear_4.view(1, 196, 3, 16, -1)
        linear_4 = None
        qkv_1 = view_9.permute(2, 0, 3, 1, 4)
        view_9 = None
        reshape_9 = qkv_1.reshape(3, 16, 196, -1)
        qkv_1 = None
        unbind_1 = reshape_9.unbind(0)
        reshape_9 = None
        q_1 = unbind_1[0]
        k_1 = unbind_1[1]
        v_1 = unbind_1[2]
        unbind_1 = None
        arange_4 = torch.arange(14)
        getitem_19 = arange_4[(slice(None, None, None), None)]
        arange_4 = None
        q_coords_2 = getitem_19 * 1.0
        getitem_19 = None
        arange_5 = torch.arange(14)
        getitem_20 = arange_5[(None, slice(None, None, None))]
        arange_5 = None
        k_coords_2 = getitem_20 * 1.0
        getitem_20 = None
        sub_2 = q_coords_2 - k_coords_2
        q_coords_2 = k_coords_2 = None
        relative_coords_2 = sub_2 + 13.0
        sub_2 = None
        long_2 = relative_coords_2.long()
        relative_coords_2 = None
        Rh_1 = l_self_modules_blocks_modules_1_modules_attn_parameters_rel_pos_h_[
            long_2
        ]
        l_self_modules_blocks_modules_1_modules_attn_parameters_rel_pos_h_ = (
            long_2
        ) = None
        arange_6 = torch.arange(14)
        getitem_22 = arange_6[(slice(None, None, None), None)]
        arange_6 = None
        q_coords_3 = getitem_22 * 1.0
        getitem_22 = None
        arange_7 = torch.arange(14)
        getitem_23 = arange_7[(None, slice(None, None, None))]
        arange_7 = None
        k_coords_3 = getitem_23 * 1.0
        getitem_23 = None
        sub_3 = q_coords_3 - k_coords_3
        q_coords_3 = k_coords_3 = None
        relative_coords_3 = sub_3 + 13.0
        sub_3 = None
        long_3 = relative_coords_3.long()
        relative_coords_3 = None
        Rw_1 = l_self_modules_blocks_modules_1_modules_attn_parameters_rel_pos_w_[
            long_3
        ]
        l_self_modules_blocks_modules_1_modules_attn_parameters_rel_pos_w_ = (
            long_3
        ) = None
        r_q_1 = q_1.reshape(16, 14, 14, 80)
        rel_h_1 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_1, Rh_1)
        Rh_1 = None
        rel_w_1 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_1, Rw_1)
        r_q_1 = Rw_1 = None
        getitem_25 = rel_h_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_1 = None
        getitem_26 = rel_w_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_1 = None
        attn_bias_2 = getitem_25 + getitem_26
        getitem_25 = getitem_26 = None
        attn_bias_3 = attn_bias_2.reshape(-1, 196, 196)
        attn_bias_2 = None
        x_29 = torch._C._nn.scaled_dot_product_attention(
            q_1, k_1, v_1, attn_mask=attn_bias_3, dropout_p=0.0
        )
        q_1 = k_1 = v_1 = attn_bias_3 = None
        view_10 = x_29.view(1, 16, 196, -1)
        x_29 = None
        transpose_1 = view_10.transpose(1, 2)
        view_10 = None
        x_30 = transpose_1.reshape(1, 196, -1)
        transpose_1 = None
        x_31 = torch._C._nn.linear(
            x_30,
            l_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_30 = (
            l_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_32 = torch.nn.functional.dropout(x_31, 0.0, False, False)
        x_31 = None
        x_33 = x_32.view(1, 14, 14, -1)
        x_32 = None
        x_34 = x_33.view(1, 1, 1, 14, 14, -1)
        x_33 = None
        permute_8 = x_34.permute(0, 1, 3, 2, 4, 5)
        x_34 = None
        contiguous_4 = permute_8.contiguous()
        permute_8 = None
        x_35 = contiguous_4.view(1, 14, 14, -1)
        contiguous_4 = None
        getitem_27 = x_35[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_35 = None
        x_36 = getitem_27.contiguous()
        getitem_27 = None
        x_37 = x_24 + x_36
        x_24 = x_36 = None
        x_38 = x_37.reshape(1, 196, -1)
        x_37 = None
        layer_norm_3 = torch.nn.functional.layer_norm(
            x_38,
            (1280,),
            l_self_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_1_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_1_modules_norm2_parameters_bias_
        ) = None
        x_39 = torch._C._nn.linear(
            layer_norm_3,
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_3 = (
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_40 = torch._C._nn.gelu(x_39, approximate="none")
        x_39 = None
        x_41 = torch.nn.functional.dropout(x_40, 0.0, False, False)
        x_40 = None
        x_42 = torch._C._nn.linear(
            x_41,
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_41 = (
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_43 = torch.nn.functional.dropout(x_42, 0.0, False, False)
        x_42 = None
        x_44 = x_38 + x_43
        x_38 = x_43 = None
        x_45 = x_44.reshape(1, 14, 14, -1)
        x_44 = None
        x_46 = torch.nn.functional.layer_norm(
            x_45,
            (1280,),
            l_self_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_2_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_2_modules_norm1_parameters_bias_
        ) = None
        x_47 = torch._C._nn.pad(x_46, (0, 0, 0, 0, 0, 0), "constant", None)
        x_46 = None
        x_48 = x_47.view(1, 1, 14, 1, 14, 1280)
        x_47 = None
        permute_9 = x_48.permute(0, 1, 3, 2, 4, 5)
        x_48 = None
        contiguous_6 = permute_9.contiguous()
        permute_9 = None
        windows_2 = contiguous_6.view(-1, 14, 14, 1280)
        contiguous_6 = None
        x_49 = windows_2.reshape(1, 196, -1)
        windows_2 = None
        linear_8 = torch._C._nn.linear(
            x_49,
            l_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_,
        )
        x_49 = (
            l_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_16 = linear_8.view(1, 196, 3, 16, -1)
        linear_8 = None
        qkv_2 = view_16.permute(2, 0, 3, 1, 4)
        view_16 = None
        reshape_16 = qkv_2.reshape(3, 16, 196, -1)
        qkv_2 = None
        unbind_2 = reshape_16.unbind(0)
        reshape_16 = None
        q_2 = unbind_2[0]
        k_2 = unbind_2[1]
        v_2 = unbind_2[2]
        unbind_2 = None
        arange_8 = torch.arange(14)
        getitem_31 = arange_8[(slice(None, None, None), None)]
        arange_8 = None
        q_coords_4 = getitem_31 * 1.0
        getitem_31 = None
        arange_9 = torch.arange(14)
        getitem_32 = arange_9[(None, slice(None, None, None))]
        arange_9 = None
        k_coords_4 = getitem_32 * 1.0
        getitem_32 = None
        sub_4 = q_coords_4 - k_coords_4
        q_coords_4 = k_coords_4 = None
        relative_coords_4 = sub_4 + 13.0
        sub_4 = None
        long_4 = relative_coords_4.long()
        relative_coords_4 = None
        Rh_2 = l_self_modules_blocks_modules_2_modules_attn_parameters_rel_pos_h_[
            long_4
        ]
        l_self_modules_blocks_modules_2_modules_attn_parameters_rel_pos_h_ = (
            long_4
        ) = None
        arange_10 = torch.arange(14)
        getitem_34 = arange_10[(slice(None, None, None), None)]
        arange_10 = None
        q_coords_5 = getitem_34 * 1.0
        getitem_34 = None
        arange_11 = torch.arange(14)
        getitem_35 = arange_11[(None, slice(None, None, None))]
        arange_11 = None
        k_coords_5 = getitem_35 * 1.0
        getitem_35 = None
        sub_5 = q_coords_5 - k_coords_5
        q_coords_5 = k_coords_5 = None
        relative_coords_5 = sub_5 + 13.0
        sub_5 = None
        long_5 = relative_coords_5.long()
        relative_coords_5 = None
        Rw_2 = l_self_modules_blocks_modules_2_modules_attn_parameters_rel_pos_w_[
            long_5
        ]
        l_self_modules_blocks_modules_2_modules_attn_parameters_rel_pos_w_ = (
            long_5
        ) = None
        r_q_2 = q_2.reshape(16, 14, 14, 80)
        rel_h_2 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_2, Rh_2)
        Rh_2 = None
        rel_w_2 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_2, Rw_2)
        r_q_2 = Rw_2 = None
        getitem_37 = rel_h_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_2 = None
        getitem_38 = rel_w_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_2 = None
        attn_bias_4 = getitem_37 + getitem_38
        getitem_37 = getitem_38 = None
        attn_bias_5 = attn_bias_4.reshape(-1, 196, 196)
        attn_bias_4 = None
        x_50 = torch._C._nn.scaled_dot_product_attention(
            q_2, k_2, v_2, attn_mask=attn_bias_5, dropout_p=0.0
        )
        q_2 = k_2 = v_2 = attn_bias_5 = None
        view_17 = x_50.view(1, 16, 196, -1)
        x_50 = None
        transpose_2 = view_17.transpose(1, 2)
        view_17 = None
        x_51 = transpose_2.reshape(1, 196, -1)
        transpose_2 = None
        x_52 = torch._C._nn.linear(
            x_51,
            l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_,
        )
        x_51 = (
            l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_53 = torch.nn.functional.dropout(x_52, 0.0, False, False)
        x_52 = None
        x_54 = x_53.view(1, 14, 14, -1)
        x_53 = None
        x_55 = x_54.view(1, 1, 1, 14, 14, -1)
        x_54 = None
        permute_11 = x_55.permute(0, 1, 3, 2, 4, 5)
        x_55 = None
        contiguous_7 = permute_11.contiguous()
        permute_11 = None
        x_56 = contiguous_7.view(1, 14, 14, -1)
        contiguous_7 = None
        getitem_39 = x_56[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_56 = None
        x_57 = getitem_39.contiguous()
        getitem_39 = None
        x_58 = x_45 + x_57
        x_45 = x_57 = None
        x_59 = x_58.reshape(1, 196, -1)
        x_58 = None
        layer_norm_5 = torch.nn.functional.layer_norm(
            x_59,
            (1280,),
            l_self_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_2_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_2_modules_norm2_parameters_bias_
        ) = None
        x_60 = torch._C._nn.linear(
            layer_norm_5,
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_5 = (
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_61 = torch._C._nn.gelu(x_60, approximate="none")
        x_60 = None
        x_62 = torch.nn.functional.dropout(x_61, 0.0, False, False)
        x_61 = None
        x_63 = torch._C._nn.linear(
            x_62,
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_62 = (
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_64 = torch.nn.functional.dropout(x_63, 0.0, False, False)
        x_63 = None
        x_65 = x_59 + x_64
        x_59 = x_64 = None
        x_66 = x_65.reshape(1, 14, 14, -1)
        x_65 = None
        x_67 = torch.nn.functional.layer_norm(
            x_66,
            (1280,),
            l_self_modules_blocks_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_3_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_3_modules_norm1_parameters_bias_
        ) = None
        x_68 = torch._C._nn.pad(x_67, (0, 0, 0, 0, 0, 0), "constant", None)
        x_67 = None
        x_69 = x_68.view(1, 1, 14, 1, 14, 1280)
        x_68 = None
        permute_12 = x_69.permute(0, 1, 3, 2, 4, 5)
        x_69 = None
        contiguous_9 = permute_12.contiguous()
        permute_12 = None
        windows_3 = contiguous_9.view(-1, 14, 14, 1280)
        contiguous_9 = None
        x_70 = windows_3.reshape(1, 196, -1)
        windows_3 = None
        linear_12 = torch._C._nn.linear(
            x_70,
            l_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_bias_,
        )
        x_70 = (
            l_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_23 = linear_12.view(1, 196, 3, 16, -1)
        linear_12 = None
        qkv_3 = view_23.permute(2, 0, 3, 1, 4)
        view_23 = None
        reshape_23 = qkv_3.reshape(3, 16, 196, -1)
        qkv_3 = None
        unbind_3 = reshape_23.unbind(0)
        reshape_23 = None
        q_3 = unbind_3[0]
        k_3 = unbind_3[1]
        v_3 = unbind_3[2]
        unbind_3 = None
        arange_12 = torch.arange(14)
        getitem_43 = arange_12[(slice(None, None, None), None)]
        arange_12 = None
        q_coords_6 = getitem_43 * 1.0
        getitem_43 = None
        arange_13 = torch.arange(14)
        getitem_44 = arange_13[(None, slice(None, None, None))]
        arange_13 = None
        k_coords_6 = getitem_44 * 1.0
        getitem_44 = None
        sub_6 = q_coords_6 - k_coords_6
        q_coords_6 = k_coords_6 = None
        relative_coords_6 = sub_6 + 13.0
        sub_6 = None
        long_6 = relative_coords_6.long()
        relative_coords_6 = None
        Rh_3 = l_self_modules_blocks_modules_3_modules_attn_parameters_rel_pos_h_[
            long_6
        ]
        l_self_modules_blocks_modules_3_modules_attn_parameters_rel_pos_h_ = (
            long_6
        ) = None
        arange_14 = torch.arange(14)
        getitem_46 = arange_14[(slice(None, None, None), None)]
        arange_14 = None
        q_coords_7 = getitem_46 * 1.0
        getitem_46 = None
        arange_15 = torch.arange(14)
        getitem_47 = arange_15[(None, slice(None, None, None))]
        arange_15 = None
        k_coords_7 = getitem_47 * 1.0
        getitem_47 = None
        sub_7 = q_coords_7 - k_coords_7
        q_coords_7 = k_coords_7 = None
        relative_coords_7 = sub_7 + 13.0
        sub_7 = None
        long_7 = relative_coords_7.long()
        relative_coords_7 = None
        Rw_3 = l_self_modules_blocks_modules_3_modules_attn_parameters_rel_pos_w_[
            long_7
        ]
        l_self_modules_blocks_modules_3_modules_attn_parameters_rel_pos_w_ = (
            long_7
        ) = None
        r_q_3 = q_3.reshape(16, 14, 14, 80)
        rel_h_3 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_3, Rh_3)
        Rh_3 = None
        rel_w_3 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_3, Rw_3)
        r_q_3 = Rw_3 = None
        getitem_49 = rel_h_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_3 = None
        getitem_50 = rel_w_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_3 = None
        attn_bias_6 = getitem_49 + getitem_50
        getitem_49 = getitem_50 = None
        attn_bias_7 = attn_bias_6.reshape(-1, 196, 196)
        attn_bias_6 = None
        x_71 = torch._C._nn.scaled_dot_product_attention(
            q_3, k_3, v_3, attn_mask=attn_bias_7, dropout_p=0.0
        )
        q_3 = k_3 = v_3 = attn_bias_7 = None
        view_24 = x_71.view(1, 16, 196, -1)
        x_71 = None
        transpose_3 = view_24.transpose(1, 2)
        view_24 = None
        x_72 = transpose_3.reshape(1, 196, -1)
        transpose_3 = None
        x_73 = torch._C._nn.linear(
            x_72,
            l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_,
        )
        x_72 = (
            l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_74 = torch.nn.functional.dropout(x_73, 0.0, False, False)
        x_73 = None
        x_75 = x_74.view(1, 14, 14, -1)
        x_74 = None
        x_76 = x_75.view(1, 1, 1, 14, 14, -1)
        x_75 = None
        permute_14 = x_76.permute(0, 1, 3, 2, 4, 5)
        x_76 = None
        contiguous_10 = permute_14.contiguous()
        permute_14 = None
        x_77 = contiguous_10.view(1, 14, 14, -1)
        contiguous_10 = None
        getitem_51 = x_77[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_77 = None
        x_78 = getitem_51.contiguous()
        getitem_51 = None
        x_79 = x_66 + x_78
        x_66 = x_78 = None
        x_80 = x_79.reshape(1, 196, -1)
        x_79 = None
        layer_norm_7 = torch.nn.functional.layer_norm(
            x_80,
            (1280,),
            l_self_modules_blocks_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_3_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_3_modules_norm2_parameters_bias_
        ) = None
        x_81 = torch._C._nn.linear(
            layer_norm_7,
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_7 = (
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_82 = torch._C._nn.gelu(x_81, approximate="none")
        x_81 = None
        x_83 = torch.nn.functional.dropout(x_82, 0.0, False, False)
        x_82 = None
        x_84 = torch._C._nn.linear(
            x_83,
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_83 = (
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_85 = torch.nn.functional.dropout(x_84, 0.0, False, False)
        x_84 = None
        x_86 = x_80 + x_85
        x_80 = x_85 = None
        x_87 = x_86.reshape(1, 14, 14, -1)
        x_86 = None
        x_88 = torch.nn.functional.layer_norm(
            x_87,
            (1280,),
            l_self_modules_blocks_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_4_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_4_modules_norm1_parameters_bias_
        ) = None
        x_89 = torch._C._nn.pad(x_88, (0, 0, 0, 0, 0, 0), "constant", None)
        x_88 = None
        x_90 = x_89.view(1, 1, 14, 1, 14, 1280)
        x_89 = None
        permute_15 = x_90.permute(0, 1, 3, 2, 4, 5)
        x_90 = None
        contiguous_12 = permute_15.contiguous()
        permute_15 = None
        windows_4 = contiguous_12.view(-1, 14, 14, 1280)
        contiguous_12 = None
        x_91 = windows_4.reshape(1, 196, -1)
        windows_4 = None
        linear_16 = torch._C._nn.linear(
            x_91,
            l_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_bias_,
        )
        x_91 = (
            l_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_30 = linear_16.view(1, 196, 3, 16, -1)
        linear_16 = None
        qkv_4 = view_30.permute(2, 0, 3, 1, 4)
        view_30 = None
        reshape_30 = qkv_4.reshape(3, 16, 196, -1)
        qkv_4 = None
        unbind_4 = reshape_30.unbind(0)
        reshape_30 = None
        q_4 = unbind_4[0]
        k_4 = unbind_4[1]
        v_4 = unbind_4[2]
        unbind_4 = None
        arange_16 = torch.arange(14)
        getitem_55 = arange_16[(slice(None, None, None), None)]
        arange_16 = None
        q_coords_8 = getitem_55 * 1.0
        getitem_55 = None
        arange_17 = torch.arange(14)
        getitem_56 = arange_17[(None, slice(None, None, None))]
        arange_17 = None
        k_coords_8 = getitem_56 * 1.0
        getitem_56 = None
        sub_8 = q_coords_8 - k_coords_8
        q_coords_8 = k_coords_8 = None
        relative_coords_8 = sub_8 + 13.0
        sub_8 = None
        long_8 = relative_coords_8.long()
        relative_coords_8 = None
        Rh_4 = l_self_modules_blocks_modules_4_modules_attn_parameters_rel_pos_h_[
            long_8
        ]
        l_self_modules_blocks_modules_4_modules_attn_parameters_rel_pos_h_ = (
            long_8
        ) = None
        arange_18 = torch.arange(14)
        getitem_58 = arange_18[(slice(None, None, None), None)]
        arange_18 = None
        q_coords_9 = getitem_58 * 1.0
        getitem_58 = None
        arange_19 = torch.arange(14)
        getitem_59 = arange_19[(None, slice(None, None, None))]
        arange_19 = None
        k_coords_9 = getitem_59 * 1.0
        getitem_59 = None
        sub_9 = q_coords_9 - k_coords_9
        q_coords_9 = k_coords_9 = None
        relative_coords_9 = sub_9 + 13.0
        sub_9 = None
        long_9 = relative_coords_9.long()
        relative_coords_9 = None
        Rw_4 = l_self_modules_blocks_modules_4_modules_attn_parameters_rel_pos_w_[
            long_9
        ]
        l_self_modules_blocks_modules_4_modules_attn_parameters_rel_pos_w_ = (
            long_9
        ) = None
        r_q_4 = q_4.reshape(16, 14, 14, 80)
        rel_h_4 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_4, Rh_4)
        Rh_4 = None
        rel_w_4 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_4, Rw_4)
        r_q_4 = Rw_4 = None
        getitem_61 = rel_h_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_4 = None
        getitem_62 = rel_w_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_4 = None
        attn_bias_8 = getitem_61 + getitem_62
        getitem_61 = getitem_62 = None
        attn_bias_9 = attn_bias_8.reshape(-1, 196, 196)
        attn_bias_8 = None
        x_92 = torch._C._nn.scaled_dot_product_attention(
            q_4, k_4, v_4, attn_mask=attn_bias_9, dropout_p=0.0
        )
        q_4 = k_4 = v_4 = attn_bias_9 = None
        view_31 = x_92.view(1, 16, 196, -1)
        x_92 = None
        transpose_4 = view_31.transpose(1, 2)
        view_31 = None
        x_93 = transpose_4.reshape(1, 196, -1)
        transpose_4 = None
        x_94 = torch._C._nn.linear(
            x_93,
            l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_,
        )
        x_93 = (
            l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_95 = torch.nn.functional.dropout(x_94, 0.0, False, False)
        x_94 = None
        x_96 = x_95.view(1, 14, 14, -1)
        x_95 = None
        x_97 = x_96.view(1, 1, 1, 14, 14, -1)
        x_96 = None
        permute_17 = x_97.permute(0, 1, 3, 2, 4, 5)
        x_97 = None
        contiguous_13 = permute_17.contiguous()
        permute_17 = None
        x_98 = contiguous_13.view(1, 14, 14, -1)
        contiguous_13 = None
        getitem_63 = x_98[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_98 = None
        x_99 = getitem_63.contiguous()
        getitem_63 = None
        x_100 = x_87 + x_99
        x_87 = x_99 = None
        x_101 = x_100.reshape(1, 196, -1)
        x_100 = None
        layer_norm_9 = torch.nn.functional.layer_norm(
            x_101,
            (1280,),
            l_self_modules_blocks_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_4_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_4_modules_norm2_parameters_bias_
        ) = None
        x_102 = torch._C._nn.linear(
            layer_norm_9,
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_9 = (
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_103 = torch._C._nn.gelu(x_102, approximate="none")
        x_102 = None
        x_104 = torch.nn.functional.dropout(x_103, 0.0, False, False)
        x_103 = None
        x_105 = torch._C._nn.linear(
            x_104,
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_104 = (
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_106 = torch.nn.functional.dropout(x_105, 0.0, False, False)
        x_105 = None
        x_107 = x_101 + x_106
        x_101 = x_106 = None
        x_108 = x_107.reshape(1, 14, 14, -1)
        x_107 = None
        x_109 = torch.nn.functional.layer_norm(
            x_108,
            (1280,),
            l_self_modules_blocks_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_5_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_5_modules_norm1_parameters_bias_
        ) = None
        x_110 = torch._C._nn.pad(x_109, (0, 0, 0, 0, 0, 0), "constant", None)
        x_109 = None
        x_111 = x_110.view(1, 1, 14, 1, 14, 1280)
        x_110 = None
        permute_18 = x_111.permute(0, 1, 3, 2, 4, 5)
        x_111 = None
        contiguous_15 = permute_18.contiguous()
        permute_18 = None
        windows_5 = contiguous_15.view(-1, 14, 14, 1280)
        contiguous_15 = None
        x_112 = windows_5.reshape(1, 196, -1)
        windows_5 = None
        linear_20 = torch._C._nn.linear(
            x_112,
            l_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_bias_,
        )
        x_112 = (
            l_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_37 = linear_20.view(1, 196, 3, 16, -1)
        linear_20 = None
        qkv_5 = view_37.permute(2, 0, 3, 1, 4)
        view_37 = None
        reshape_37 = qkv_5.reshape(3, 16, 196, -1)
        qkv_5 = None
        unbind_5 = reshape_37.unbind(0)
        reshape_37 = None
        q_5 = unbind_5[0]
        k_5 = unbind_5[1]
        v_5 = unbind_5[2]
        unbind_5 = None
        arange_20 = torch.arange(14)
        getitem_67 = arange_20[(slice(None, None, None), None)]
        arange_20 = None
        q_coords_10 = getitem_67 * 1.0
        getitem_67 = None
        arange_21 = torch.arange(14)
        getitem_68 = arange_21[(None, slice(None, None, None))]
        arange_21 = None
        k_coords_10 = getitem_68 * 1.0
        getitem_68 = None
        sub_10 = q_coords_10 - k_coords_10
        q_coords_10 = k_coords_10 = None
        relative_coords_10 = sub_10 + 13.0
        sub_10 = None
        long_10 = relative_coords_10.long()
        relative_coords_10 = None
        Rh_5 = l_self_modules_blocks_modules_5_modules_attn_parameters_rel_pos_h_[
            long_10
        ]
        l_self_modules_blocks_modules_5_modules_attn_parameters_rel_pos_h_ = (
            long_10
        ) = None
        arange_22 = torch.arange(14)
        getitem_70 = arange_22[(slice(None, None, None), None)]
        arange_22 = None
        q_coords_11 = getitem_70 * 1.0
        getitem_70 = None
        arange_23 = torch.arange(14)
        getitem_71 = arange_23[(None, slice(None, None, None))]
        arange_23 = None
        k_coords_11 = getitem_71 * 1.0
        getitem_71 = None
        sub_11 = q_coords_11 - k_coords_11
        q_coords_11 = k_coords_11 = None
        relative_coords_11 = sub_11 + 13.0
        sub_11 = None
        long_11 = relative_coords_11.long()
        relative_coords_11 = None
        Rw_5 = l_self_modules_blocks_modules_5_modules_attn_parameters_rel_pos_w_[
            long_11
        ]
        l_self_modules_blocks_modules_5_modules_attn_parameters_rel_pos_w_ = (
            long_11
        ) = None
        r_q_5 = q_5.reshape(16, 14, 14, 80)
        rel_h_5 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_5, Rh_5)
        Rh_5 = None
        rel_w_5 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_5, Rw_5)
        r_q_5 = Rw_5 = None
        getitem_73 = rel_h_5[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_5 = None
        getitem_74 = rel_w_5[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_5 = None
        attn_bias_10 = getitem_73 + getitem_74
        getitem_73 = getitem_74 = None
        attn_bias_11 = attn_bias_10.reshape(-1, 196, 196)
        attn_bias_10 = None
        x_113 = torch._C._nn.scaled_dot_product_attention(
            q_5, k_5, v_5, attn_mask=attn_bias_11, dropout_p=0.0
        )
        q_5 = k_5 = v_5 = attn_bias_11 = None
        view_38 = x_113.view(1, 16, 196, -1)
        x_113 = None
        transpose_5 = view_38.transpose(1, 2)
        view_38 = None
        x_114 = transpose_5.reshape(1, 196, -1)
        transpose_5 = None
        x_115 = torch._C._nn.linear(
            x_114,
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_,
        )
        x_114 = (
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_116 = torch.nn.functional.dropout(x_115, 0.0, False, False)
        x_115 = None
        x_117 = x_116.view(1, 14, 14, -1)
        x_116 = None
        x_118 = x_117.view(1, 1, 1, 14, 14, -1)
        x_117 = None
        permute_20 = x_118.permute(0, 1, 3, 2, 4, 5)
        x_118 = None
        contiguous_16 = permute_20.contiguous()
        permute_20 = None
        x_119 = contiguous_16.view(1, 14, 14, -1)
        contiguous_16 = None
        getitem_75 = x_119[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_119 = None
        x_120 = getitem_75.contiguous()
        getitem_75 = None
        x_121 = x_108 + x_120
        x_108 = x_120 = None
        x_122 = x_121.reshape(1, 196, -1)
        x_121 = None
        layer_norm_11 = torch.nn.functional.layer_norm(
            x_122,
            (1280,),
            l_self_modules_blocks_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_5_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_5_modules_norm2_parameters_bias_
        ) = None
        x_123 = torch._C._nn.linear(
            layer_norm_11,
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_11 = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_124 = torch._C._nn.gelu(x_123, approximate="none")
        x_123 = None
        x_125 = torch.nn.functional.dropout(x_124, 0.0, False, False)
        x_124 = None
        x_126 = torch._C._nn.linear(
            x_125,
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_125 = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_127 = torch.nn.functional.dropout(x_126, 0.0, False, False)
        x_126 = None
        x_128 = x_122 + x_127
        x_122 = x_127 = None
        x_129 = x_128.reshape(1, 14, 14, -1)
        x_128 = None
        x_130 = torch.nn.functional.layer_norm(
            x_129,
            (1280,),
            l_self_modules_blocks_modules_6_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_6_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_6_modules_norm1_parameters_bias_
        ) = None
        x_131 = torch._C._nn.pad(x_130, (0, 0, 0, 0, 0, 0), "constant", None)
        x_130 = None
        x_132 = x_131.view(1, 1, 14, 1, 14, 1280)
        x_131 = None
        permute_21 = x_132.permute(0, 1, 3, 2, 4, 5)
        x_132 = None
        contiguous_18 = permute_21.contiguous()
        permute_21 = None
        windows_6 = contiguous_18.view(-1, 14, 14, 1280)
        contiguous_18 = None
        x_133 = windows_6.reshape(1, 196, -1)
        windows_6 = None
        linear_24 = torch._C._nn.linear(
            x_133,
            l_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_bias_,
        )
        x_133 = (
            l_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_44 = linear_24.view(1, 196, 3, 16, -1)
        linear_24 = None
        qkv_6 = view_44.permute(2, 0, 3, 1, 4)
        view_44 = None
        reshape_44 = qkv_6.reshape(3, 16, 196, -1)
        qkv_6 = None
        unbind_6 = reshape_44.unbind(0)
        reshape_44 = None
        q_6 = unbind_6[0]
        k_6 = unbind_6[1]
        v_6 = unbind_6[2]
        unbind_6 = None
        arange_24 = torch.arange(14)
        getitem_79 = arange_24[(slice(None, None, None), None)]
        arange_24 = None
        q_coords_12 = getitem_79 * 1.0
        getitem_79 = None
        arange_25 = torch.arange(14)
        getitem_80 = arange_25[(None, slice(None, None, None))]
        arange_25 = None
        k_coords_12 = getitem_80 * 1.0
        getitem_80 = None
        sub_12 = q_coords_12 - k_coords_12
        q_coords_12 = k_coords_12 = None
        relative_coords_12 = sub_12 + 13.0
        sub_12 = None
        long_12 = relative_coords_12.long()
        relative_coords_12 = None
        Rh_6 = l_self_modules_blocks_modules_6_modules_attn_parameters_rel_pos_h_[
            long_12
        ]
        l_self_modules_blocks_modules_6_modules_attn_parameters_rel_pos_h_ = (
            long_12
        ) = None
        arange_26 = torch.arange(14)
        getitem_82 = arange_26[(slice(None, None, None), None)]
        arange_26 = None
        q_coords_13 = getitem_82 * 1.0
        getitem_82 = None
        arange_27 = torch.arange(14)
        getitem_83 = arange_27[(None, slice(None, None, None))]
        arange_27 = None
        k_coords_13 = getitem_83 * 1.0
        getitem_83 = None
        sub_13 = q_coords_13 - k_coords_13
        q_coords_13 = k_coords_13 = None
        relative_coords_13 = sub_13 + 13.0
        sub_13 = None
        long_13 = relative_coords_13.long()
        relative_coords_13 = None
        Rw_6 = l_self_modules_blocks_modules_6_modules_attn_parameters_rel_pos_w_[
            long_13
        ]
        l_self_modules_blocks_modules_6_modules_attn_parameters_rel_pos_w_ = (
            long_13
        ) = None
        r_q_6 = q_6.reshape(16, 14, 14, 80)
        rel_h_6 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_6, Rh_6)
        Rh_6 = None
        rel_w_6 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_6, Rw_6)
        r_q_6 = Rw_6 = None
        getitem_85 = rel_h_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_6 = None
        getitem_86 = rel_w_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_6 = None
        attn_bias_12 = getitem_85 + getitem_86
        getitem_85 = getitem_86 = None
        attn_bias_13 = attn_bias_12.reshape(-1, 196, 196)
        attn_bias_12 = None
        x_134 = torch._C._nn.scaled_dot_product_attention(
            q_6, k_6, v_6, attn_mask=attn_bias_13, dropout_p=0.0
        )
        q_6 = k_6 = v_6 = attn_bias_13 = None
        view_45 = x_134.view(1, 16, 196, -1)
        x_134 = None
        transpose_6 = view_45.transpose(1, 2)
        view_45 = None
        x_135 = transpose_6.reshape(1, 196, -1)
        transpose_6 = None
        x_136 = torch._C._nn.linear(
            x_135,
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_,
        )
        x_135 = (
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_137 = torch.nn.functional.dropout(x_136, 0.0, False, False)
        x_136 = None
        x_138 = x_137.view(1, 14, 14, -1)
        x_137 = None
        x_139 = x_138.view(1, 1, 1, 14, 14, -1)
        x_138 = None
        permute_23 = x_139.permute(0, 1, 3, 2, 4, 5)
        x_139 = None
        contiguous_19 = permute_23.contiguous()
        permute_23 = None
        x_140 = contiguous_19.view(1, 14, 14, -1)
        contiguous_19 = None
        getitem_87 = x_140[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_140 = None
        x_141 = getitem_87.contiguous()
        getitem_87 = None
        x_142 = x_129 + x_141
        x_129 = x_141 = None
        x_143 = x_142.reshape(1, 196, -1)
        x_142 = None
        layer_norm_13 = torch.nn.functional.layer_norm(
            x_143,
            (1280,),
            l_self_modules_blocks_modules_6_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_6_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_6_modules_norm2_parameters_bias_
        ) = None
        x_144 = torch._C._nn.linear(
            layer_norm_13,
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_13 = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_145 = torch._C._nn.gelu(x_144, approximate="none")
        x_144 = None
        x_146 = torch.nn.functional.dropout(x_145, 0.0, False, False)
        x_145 = None
        x_147 = torch._C._nn.linear(
            x_146,
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_146 = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_148 = torch.nn.functional.dropout(x_147, 0.0, False, False)
        x_147 = None
        x_149 = x_143 + x_148
        x_143 = x_148 = None
        x_150 = x_149.reshape(1, 14, 14, -1)
        x_149 = None
        x_151 = torch.nn.functional.layer_norm(
            x_150,
            (1280,),
            l_self_modules_blocks_modules_7_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_7_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_7_modules_norm1_parameters_bias_
        ) = None
        x_152 = x_151.reshape(1, 196, -1)
        x_151 = None
        linear_28 = torch._C._nn.linear(
            x_152,
            l_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_bias_,
        )
        x_152 = (
            l_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_49 = linear_28.view(1, 196, 3, 16, -1)
        linear_28 = None
        qkv_7 = view_49.permute(2, 0, 3, 1, 4)
        view_49 = None
        reshape_51 = qkv_7.reshape(3, 16, 196, -1)
        qkv_7 = None
        unbind_7 = reshape_51.unbind(0)
        reshape_51 = None
        q_7 = unbind_7[0]
        k_7 = unbind_7[1]
        v_7 = unbind_7[2]
        unbind_7 = None
        reshape_52 = (
            l_self_modules_blocks_modules_7_modules_attn_parameters_rel_pos_h_.reshape(
                1, 127, -1
            )
        )
        l_self_modules_blocks_modules_7_modules_attn_parameters_rel_pos_h_ = None
        permute_25 = reshape_52.permute(0, 2, 1)
        reshape_52 = None
        rel_pos_resized = torch.nn.functional.interpolate(
            permute_25, size=27, mode="linear"
        )
        permute_25 = None
        reshape_53 = rel_pos_resized.reshape(-1, 27)
        rel_pos_resized = None
        rel_pos_resized_1 = reshape_53.permute(1, 0)
        reshape_53 = None
        arange_28 = torch.arange(14)
        getitem_91 = arange_28[(slice(None, None, None), None)]
        arange_28 = None
        q_coords_14 = getitem_91 * 1.0
        getitem_91 = None
        arange_29 = torch.arange(14)
        getitem_92 = arange_29[(None, slice(None, None, None))]
        arange_29 = None
        k_coords_14 = getitem_92 * 1.0
        getitem_92 = None
        sub_14 = q_coords_14 - k_coords_14
        q_coords_14 = k_coords_14 = None
        relative_coords_14 = sub_14 + 13.0
        sub_14 = None
        long_14 = relative_coords_14.long()
        relative_coords_14 = None
        Rh_7 = rel_pos_resized_1[long_14]
        rel_pos_resized_1 = long_14 = None
        reshape_54 = (
            l_self_modules_blocks_modules_7_modules_attn_parameters_rel_pos_w_.reshape(
                1, 127, -1
            )
        )
        l_self_modules_blocks_modules_7_modules_attn_parameters_rel_pos_w_ = None
        permute_27 = reshape_54.permute(0, 2, 1)
        reshape_54 = None
        rel_pos_resized_2 = torch.nn.functional.interpolate(
            permute_27, size=27, mode="linear"
        )
        permute_27 = None
        reshape_55 = rel_pos_resized_2.reshape(-1, 27)
        rel_pos_resized_2 = None
        rel_pos_resized_3 = reshape_55.permute(1, 0)
        reshape_55 = None
        arange_30 = torch.arange(14)
        getitem_94 = arange_30[(slice(None, None, None), None)]
        arange_30 = None
        q_coords_15 = getitem_94 * 1.0
        getitem_94 = None
        arange_31 = torch.arange(14)
        getitem_95 = arange_31[(None, slice(None, None, None))]
        arange_31 = None
        k_coords_15 = getitem_95 * 1.0
        getitem_95 = None
        sub_15 = q_coords_15 - k_coords_15
        q_coords_15 = k_coords_15 = None
        relative_coords_15 = sub_15 + 13.0
        sub_15 = None
        long_15 = relative_coords_15.long()
        relative_coords_15 = None
        Rw_7 = rel_pos_resized_3[long_15]
        rel_pos_resized_3 = long_15 = None
        r_q_7 = q_7.reshape(16, 14, 14, 80)
        rel_h_7 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_7, Rh_7)
        Rh_7 = None
        rel_w_7 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_7, Rw_7)
        r_q_7 = Rw_7 = None
        getitem_97 = rel_h_7[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_7 = None
        getitem_98 = rel_w_7[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_7 = None
        attn_bias_14 = getitem_97 + getitem_98
        getitem_97 = getitem_98 = None
        attn_bias_15 = attn_bias_14.reshape(-1, 196, 196)
        attn_bias_14 = None
        x_153 = torch._C._nn.scaled_dot_product_attention(
            q_7, k_7, v_7, attn_mask=attn_bias_15, dropout_p=0.0
        )
        q_7 = k_7 = v_7 = attn_bias_15 = None
        view_50 = x_153.view(1, 16, 196, -1)
        x_153 = None
        transpose_7 = view_50.transpose(1, 2)
        view_50 = None
        x_154 = transpose_7.reshape(1, 196, -1)
        transpose_7 = None
        x_155 = torch._C._nn.linear(
            x_154,
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_,
        )
        x_154 = (
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_156 = torch.nn.functional.dropout(x_155, 0.0, False, False)
        x_155 = None
        x_157 = x_156.view(1, 14, 14, -1)
        x_156 = None
        x_158 = x_150 + x_157
        x_150 = x_157 = None
        x_159 = x_158.reshape(1, 196, -1)
        x_158 = None
        layer_norm_15 = torch.nn.functional.layer_norm(
            x_159,
            (1280,),
            l_self_modules_blocks_modules_7_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_7_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_7_modules_norm2_parameters_bias_
        ) = None
        x_160 = torch._C._nn.linear(
            layer_norm_15,
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_15 = (
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_161 = torch._C._nn.gelu(x_160, approximate="none")
        x_160 = None
        x_162 = torch.nn.functional.dropout(x_161, 0.0, False, False)
        x_161 = None
        x_163 = torch._C._nn.linear(
            x_162,
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_162 = (
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_164 = torch.nn.functional.dropout(x_163, 0.0, False, False)
        x_163 = None
        x_165 = x_159 + x_164
        x_159 = x_164 = None
        x_166 = x_165.reshape(1, 14, 14, -1)
        x_165 = None
        x_167 = torch.nn.functional.layer_norm(
            x_166,
            (1280,),
            l_self_modules_blocks_modules_8_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_8_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_8_modules_norm1_parameters_bias_
        ) = None
        x_168 = torch._C._nn.pad(x_167, (0, 0, 0, 0, 0, 0), "constant", None)
        x_167 = None
        x_169 = x_168.view(1, 1, 14, 1, 14, 1280)
        x_168 = None
        permute_29 = x_169.permute(0, 1, 3, 2, 4, 5)
        x_169 = None
        contiguous_21 = permute_29.contiguous()
        permute_29 = None
        windows_7 = contiguous_21.view(-1, 14, 14, 1280)
        contiguous_21 = None
        x_170 = windows_7.reshape(1, 196, -1)
        windows_7 = None
        linear_32 = torch._C._nn.linear(
            x_170,
            l_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_bias_,
        )
        x_170 = (
            l_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_54 = linear_32.view(1, 196, 3, 16, -1)
        linear_32 = None
        qkv_8 = view_54.permute(2, 0, 3, 1, 4)
        view_54 = None
        reshape_62 = qkv_8.reshape(3, 16, 196, -1)
        qkv_8 = None
        unbind_8 = reshape_62.unbind(0)
        reshape_62 = None
        q_8 = unbind_8[0]
        k_8 = unbind_8[1]
        v_8 = unbind_8[2]
        unbind_8 = None
        arange_32 = torch.arange(14)
        getitem_102 = arange_32[(slice(None, None, None), None)]
        arange_32 = None
        q_coords_16 = getitem_102 * 1.0
        getitem_102 = None
        arange_33 = torch.arange(14)
        getitem_103 = arange_33[(None, slice(None, None, None))]
        arange_33 = None
        k_coords_16 = getitem_103 * 1.0
        getitem_103 = None
        sub_16 = q_coords_16 - k_coords_16
        q_coords_16 = k_coords_16 = None
        relative_coords_16 = sub_16 + 13.0
        sub_16 = None
        long_16 = relative_coords_16.long()
        relative_coords_16 = None
        Rh_8 = l_self_modules_blocks_modules_8_modules_attn_parameters_rel_pos_h_[
            long_16
        ]
        l_self_modules_blocks_modules_8_modules_attn_parameters_rel_pos_h_ = (
            long_16
        ) = None
        arange_34 = torch.arange(14)
        getitem_105 = arange_34[(slice(None, None, None), None)]
        arange_34 = None
        q_coords_17 = getitem_105 * 1.0
        getitem_105 = None
        arange_35 = torch.arange(14)
        getitem_106 = arange_35[(None, slice(None, None, None))]
        arange_35 = None
        k_coords_17 = getitem_106 * 1.0
        getitem_106 = None
        sub_17 = q_coords_17 - k_coords_17
        q_coords_17 = k_coords_17 = None
        relative_coords_17 = sub_17 + 13.0
        sub_17 = None
        long_17 = relative_coords_17.long()
        relative_coords_17 = None
        Rw_8 = l_self_modules_blocks_modules_8_modules_attn_parameters_rel_pos_w_[
            long_17
        ]
        l_self_modules_blocks_modules_8_modules_attn_parameters_rel_pos_w_ = (
            long_17
        ) = None
        r_q_8 = q_8.reshape(16, 14, 14, 80)
        rel_h_8 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_8, Rh_8)
        Rh_8 = None
        rel_w_8 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_8, Rw_8)
        r_q_8 = Rw_8 = None
        getitem_108 = rel_h_8[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_8 = None
        getitem_109 = rel_w_8[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_8 = None
        attn_bias_16 = getitem_108 + getitem_109
        getitem_108 = getitem_109 = None
        attn_bias_17 = attn_bias_16.reshape(-1, 196, 196)
        attn_bias_16 = None
        x_171 = torch._C._nn.scaled_dot_product_attention(
            q_8, k_8, v_8, attn_mask=attn_bias_17, dropout_p=0.0
        )
        q_8 = k_8 = v_8 = attn_bias_17 = None
        view_55 = x_171.view(1, 16, 196, -1)
        x_171 = None
        transpose_8 = view_55.transpose(1, 2)
        view_55 = None
        x_172 = transpose_8.reshape(1, 196, -1)
        transpose_8 = None
        x_173 = torch._C._nn.linear(
            x_172,
            l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_,
        )
        x_172 = (
            l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_174 = torch.nn.functional.dropout(x_173, 0.0, False, False)
        x_173 = None
        x_175 = x_174.view(1, 14, 14, -1)
        x_174 = None
        x_176 = x_175.view(1, 1, 1, 14, 14, -1)
        x_175 = None
        permute_31 = x_176.permute(0, 1, 3, 2, 4, 5)
        x_176 = None
        contiguous_22 = permute_31.contiguous()
        permute_31 = None
        x_177 = contiguous_22.view(1, 14, 14, -1)
        contiguous_22 = None
        getitem_110 = x_177[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_177 = None
        x_178 = getitem_110.contiguous()
        getitem_110 = None
        x_179 = x_166 + x_178
        x_166 = x_178 = None
        x_180 = x_179.reshape(1, 196, -1)
        x_179 = None
        layer_norm_17 = torch.nn.functional.layer_norm(
            x_180,
            (1280,),
            l_self_modules_blocks_modules_8_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_8_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_8_modules_norm2_parameters_bias_
        ) = None
        x_181 = torch._C._nn.linear(
            layer_norm_17,
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_17 = (
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_182 = torch._C._nn.gelu(x_181, approximate="none")
        x_181 = None
        x_183 = torch.nn.functional.dropout(x_182, 0.0, False, False)
        x_182 = None
        x_184 = torch._C._nn.linear(
            x_183,
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_183 = (
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_185 = torch.nn.functional.dropout(x_184, 0.0, False, False)
        x_184 = None
        x_186 = x_180 + x_185
        x_180 = x_185 = None
        x_187 = x_186.reshape(1, 14, 14, -1)
        x_186 = None
        x_188 = torch.nn.functional.layer_norm(
            x_187,
            (1280,),
            l_self_modules_blocks_modules_9_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_9_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_9_modules_norm1_parameters_bias_
        ) = None
        x_189 = torch._C._nn.pad(x_188, (0, 0, 0, 0, 0, 0), "constant", None)
        x_188 = None
        x_190 = x_189.view(1, 1, 14, 1, 14, 1280)
        x_189 = None
        permute_32 = x_190.permute(0, 1, 3, 2, 4, 5)
        x_190 = None
        contiguous_24 = permute_32.contiguous()
        permute_32 = None
        windows_8 = contiguous_24.view(-1, 14, 14, 1280)
        contiguous_24 = None
        x_191 = windows_8.reshape(1, 196, -1)
        windows_8 = None
        linear_36 = torch._C._nn.linear(
            x_191,
            l_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_bias_,
        )
        x_191 = (
            l_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_61 = linear_36.view(1, 196, 3, 16, -1)
        linear_36 = None
        qkv_9 = view_61.permute(2, 0, 3, 1, 4)
        view_61 = None
        reshape_69 = qkv_9.reshape(3, 16, 196, -1)
        qkv_9 = None
        unbind_9 = reshape_69.unbind(0)
        reshape_69 = None
        q_9 = unbind_9[0]
        k_9 = unbind_9[1]
        v_9 = unbind_9[2]
        unbind_9 = None
        arange_36 = torch.arange(14)
        getitem_114 = arange_36[(slice(None, None, None), None)]
        arange_36 = None
        q_coords_18 = getitem_114 * 1.0
        getitem_114 = None
        arange_37 = torch.arange(14)
        getitem_115 = arange_37[(None, slice(None, None, None))]
        arange_37 = None
        k_coords_18 = getitem_115 * 1.0
        getitem_115 = None
        sub_18 = q_coords_18 - k_coords_18
        q_coords_18 = k_coords_18 = None
        relative_coords_18 = sub_18 + 13.0
        sub_18 = None
        long_18 = relative_coords_18.long()
        relative_coords_18 = None
        Rh_9 = l_self_modules_blocks_modules_9_modules_attn_parameters_rel_pos_h_[
            long_18
        ]
        l_self_modules_blocks_modules_9_modules_attn_parameters_rel_pos_h_ = (
            long_18
        ) = None
        arange_38 = torch.arange(14)
        getitem_117 = arange_38[(slice(None, None, None), None)]
        arange_38 = None
        q_coords_19 = getitem_117 * 1.0
        getitem_117 = None
        arange_39 = torch.arange(14)
        getitem_118 = arange_39[(None, slice(None, None, None))]
        arange_39 = None
        k_coords_19 = getitem_118 * 1.0
        getitem_118 = None
        sub_19 = q_coords_19 - k_coords_19
        q_coords_19 = k_coords_19 = None
        relative_coords_19 = sub_19 + 13.0
        sub_19 = None
        long_19 = relative_coords_19.long()
        relative_coords_19 = None
        Rw_9 = l_self_modules_blocks_modules_9_modules_attn_parameters_rel_pos_w_[
            long_19
        ]
        l_self_modules_blocks_modules_9_modules_attn_parameters_rel_pos_w_ = (
            long_19
        ) = None
        r_q_9 = q_9.reshape(16, 14, 14, 80)
        rel_h_9 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_9, Rh_9)
        Rh_9 = None
        rel_w_9 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_9, Rw_9)
        r_q_9 = Rw_9 = None
        getitem_120 = rel_h_9[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_9 = None
        getitem_121 = rel_w_9[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_9 = None
        attn_bias_18 = getitem_120 + getitem_121
        getitem_120 = getitem_121 = None
        attn_bias_19 = attn_bias_18.reshape(-1, 196, 196)
        attn_bias_18 = None
        x_192 = torch._C._nn.scaled_dot_product_attention(
            q_9, k_9, v_9, attn_mask=attn_bias_19, dropout_p=0.0
        )
        q_9 = k_9 = v_9 = attn_bias_19 = None
        view_62 = x_192.view(1, 16, 196, -1)
        x_192 = None
        transpose_9 = view_62.transpose(1, 2)
        view_62 = None
        x_193 = transpose_9.reshape(1, 196, -1)
        transpose_9 = None
        x_194 = torch._C._nn.linear(
            x_193,
            l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_,
        )
        x_193 = (
            l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_195 = torch.nn.functional.dropout(x_194, 0.0, False, False)
        x_194 = None
        x_196 = x_195.view(1, 14, 14, -1)
        x_195 = None
        x_197 = x_196.view(1, 1, 1, 14, 14, -1)
        x_196 = None
        permute_34 = x_197.permute(0, 1, 3, 2, 4, 5)
        x_197 = None
        contiguous_25 = permute_34.contiguous()
        permute_34 = None
        x_198 = contiguous_25.view(1, 14, 14, -1)
        contiguous_25 = None
        getitem_122 = x_198[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_198 = None
        x_199 = getitem_122.contiguous()
        getitem_122 = None
        x_200 = x_187 + x_199
        x_187 = x_199 = None
        x_201 = x_200.reshape(1, 196, -1)
        x_200 = None
        layer_norm_19 = torch.nn.functional.layer_norm(
            x_201,
            (1280,),
            l_self_modules_blocks_modules_9_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_9_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_9_modules_norm2_parameters_bias_
        ) = None
        x_202 = torch._C._nn.linear(
            layer_norm_19,
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_19 = (
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_203 = torch._C._nn.gelu(x_202, approximate="none")
        x_202 = None
        x_204 = torch.nn.functional.dropout(x_203, 0.0, False, False)
        x_203 = None
        x_205 = torch._C._nn.linear(
            x_204,
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_204 = (
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_206 = torch.nn.functional.dropout(x_205, 0.0, False, False)
        x_205 = None
        x_207 = x_201 + x_206
        x_201 = x_206 = None
        x_208 = x_207.reshape(1, 14, 14, -1)
        x_207 = None
        x_209 = torch.nn.functional.layer_norm(
            x_208,
            (1280,),
            l_self_modules_blocks_modules_10_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_10_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_10_modules_norm1_parameters_bias_
        ) = None
        x_210 = torch._C._nn.pad(x_209, (0, 0, 0, 0, 0, 0), "constant", None)
        x_209 = None
        x_211 = x_210.view(1, 1, 14, 1, 14, 1280)
        x_210 = None
        permute_35 = x_211.permute(0, 1, 3, 2, 4, 5)
        x_211 = None
        contiguous_27 = permute_35.contiguous()
        permute_35 = None
        windows_9 = contiguous_27.view(-1, 14, 14, 1280)
        contiguous_27 = None
        x_212 = windows_9.reshape(1, 196, -1)
        windows_9 = None
        linear_40 = torch._C._nn.linear(
            x_212,
            l_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_bias_,
        )
        x_212 = (
            l_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_68 = linear_40.view(1, 196, 3, 16, -1)
        linear_40 = None
        qkv_10 = view_68.permute(2, 0, 3, 1, 4)
        view_68 = None
        reshape_76 = qkv_10.reshape(3, 16, 196, -1)
        qkv_10 = None
        unbind_10 = reshape_76.unbind(0)
        reshape_76 = None
        q_10 = unbind_10[0]
        k_10 = unbind_10[1]
        v_10 = unbind_10[2]
        unbind_10 = None
        arange_40 = torch.arange(14)
        getitem_126 = arange_40[(slice(None, None, None), None)]
        arange_40 = None
        q_coords_20 = getitem_126 * 1.0
        getitem_126 = None
        arange_41 = torch.arange(14)
        getitem_127 = arange_41[(None, slice(None, None, None))]
        arange_41 = None
        k_coords_20 = getitem_127 * 1.0
        getitem_127 = None
        sub_20 = q_coords_20 - k_coords_20
        q_coords_20 = k_coords_20 = None
        relative_coords_20 = sub_20 + 13.0
        sub_20 = None
        long_20 = relative_coords_20.long()
        relative_coords_20 = None
        Rh_10 = l_self_modules_blocks_modules_10_modules_attn_parameters_rel_pos_h_[
            long_20
        ]
        l_self_modules_blocks_modules_10_modules_attn_parameters_rel_pos_h_ = (
            long_20
        ) = None
        arange_42 = torch.arange(14)
        getitem_129 = arange_42[(slice(None, None, None), None)]
        arange_42 = None
        q_coords_21 = getitem_129 * 1.0
        getitem_129 = None
        arange_43 = torch.arange(14)
        getitem_130 = arange_43[(None, slice(None, None, None))]
        arange_43 = None
        k_coords_21 = getitem_130 * 1.0
        getitem_130 = None
        sub_21 = q_coords_21 - k_coords_21
        q_coords_21 = k_coords_21 = None
        relative_coords_21 = sub_21 + 13.0
        sub_21 = None
        long_21 = relative_coords_21.long()
        relative_coords_21 = None
        Rw_10 = l_self_modules_blocks_modules_10_modules_attn_parameters_rel_pos_w_[
            long_21
        ]
        l_self_modules_blocks_modules_10_modules_attn_parameters_rel_pos_w_ = (
            long_21
        ) = None
        r_q_10 = q_10.reshape(16, 14, 14, 80)
        rel_h_10 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_10, Rh_10)
        Rh_10 = None
        rel_w_10 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_10, Rw_10)
        r_q_10 = Rw_10 = None
        getitem_132 = rel_h_10[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_10 = None
        getitem_133 = rel_w_10[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_10 = None
        attn_bias_20 = getitem_132 + getitem_133
        getitem_132 = getitem_133 = None
        attn_bias_21 = attn_bias_20.reshape(-1, 196, 196)
        attn_bias_20 = None
        x_213 = torch._C._nn.scaled_dot_product_attention(
            q_10, k_10, v_10, attn_mask=attn_bias_21, dropout_p=0.0
        )
        q_10 = k_10 = v_10 = attn_bias_21 = None
        view_69 = x_213.view(1, 16, 196, -1)
        x_213 = None
        transpose_10 = view_69.transpose(1, 2)
        view_69 = None
        x_214 = transpose_10.reshape(1, 196, -1)
        transpose_10 = None
        x_215 = torch._C._nn.linear(
            x_214,
            l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_,
        )
        x_214 = l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_216 = torch.nn.functional.dropout(x_215, 0.0, False, False)
        x_215 = None
        x_217 = x_216.view(1, 14, 14, -1)
        x_216 = None
        x_218 = x_217.view(1, 1, 1, 14, 14, -1)
        x_217 = None
        permute_37 = x_218.permute(0, 1, 3, 2, 4, 5)
        x_218 = None
        contiguous_28 = permute_37.contiguous()
        permute_37 = None
        x_219 = contiguous_28.view(1, 14, 14, -1)
        contiguous_28 = None
        getitem_134 = x_219[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_219 = None
        x_220 = getitem_134.contiguous()
        getitem_134 = None
        x_221 = x_208 + x_220
        x_208 = x_220 = None
        x_222 = x_221.reshape(1, 196, -1)
        x_221 = None
        layer_norm_21 = torch.nn.functional.layer_norm(
            x_222,
            (1280,),
            l_self_modules_blocks_modules_10_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_10_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_10_modules_norm2_parameters_bias_
        ) = None
        x_223 = torch._C._nn.linear(
            layer_norm_21,
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_21 = (
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_224 = torch._C._nn.gelu(x_223, approximate="none")
        x_223 = None
        x_225 = torch.nn.functional.dropout(x_224, 0.0, False, False)
        x_224 = None
        x_226 = torch._C._nn.linear(
            x_225,
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_225 = (
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_227 = torch.nn.functional.dropout(x_226, 0.0, False, False)
        x_226 = None
        x_228 = x_222 + x_227
        x_222 = x_227 = None
        x_229 = x_228.reshape(1, 14, 14, -1)
        x_228 = None
        x_230 = torch.nn.functional.layer_norm(
            x_229,
            (1280,),
            l_self_modules_blocks_modules_11_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_11_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_11_modules_norm1_parameters_bias_
        ) = None
        x_231 = torch._C._nn.pad(x_230, (0, 0, 0, 0, 0, 0), "constant", None)
        x_230 = None
        x_232 = x_231.view(1, 1, 14, 1, 14, 1280)
        x_231 = None
        permute_38 = x_232.permute(0, 1, 3, 2, 4, 5)
        x_232 = None
        contiguous_30 = permute_38.contiguous()
        permute_38 = None
        windows_10 = contiguous_30.view(-1, 14, 14, 1280)
        contiguous_30 = None
        x_233 = windows_10.reshape(1, 196, -1)
        windows_10 = None
        linear_44 = torch._C._nn.linear(
            x_233,
            l_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_bias_,
        )
        x_233 = (
            l_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_75 = linear_44.view(1, 196, 3, 16, -1)
        linear_44 = None
        qkv_11 = view_75.permute(2, 0, 3, 1, 4)
        view_75 = None
        reshape_83 = qkv_11.reshape(3, 16, 196, -1)
        qkv_11 = None
        unbind_11 = reshape_83.unbind(0)
        reshape_83 = None
        q_11 = unbind_11[0]
        k_11 = unbind_11[1]
        v_11 = unbind_11[2]
        unbind_11 = None
        arange_44 = torch.arange(14)
        getitem_138 = arange_44[(slice(None, None, None), None)]
        arange_44 = None
        q_coords_22 = getitem_138 * 1.0
        getitem_138 = None
        arange_45 = torch.arange(14)
        getitem_139 = arange_45[(None, slice(None, None, None))]
        arange_45 = None
        k_coords_22 = getitem_139 * 1.0
        getitem_139 = None
        sub_22 = q_coords_22 - k_coords_22
        q_coords_22 = k_coords_22 = None
        relative_coords_22 = sub_22 + 13.0
        sub_22 = None
        long_22 = relative_coords_22.long()
        relative_coords_22 = None
        Rh_11 = l_self_modules_blocks_modules_11_modules_attn_parameters_rel_pos_h_[
            long_22
        ]
        l_self_modules_blocks_modules_11_modules_attn_parameters_rel_pos_h_ = (
            long_22
        ) = None
        arange_46 = torch.arange(14)
        getitem_141 = arange_46[(slice(None, None, None), None)]
        arange_46 = None
        q_coords_23 = getitem_141 * 1.0
        getitem_141 = None
        arange_47 = torch.arange(14)
        getitem_142 = arange_47[(None, slice(None, None, None))]
        arange_47 = None
        k_coords_23 = getitem_142 * 1.0
        getitem_142 = None
        sub_23 = q_coords_23 - k_coords_23
        q_coords_23 = k_coords_23 = None
        relative_coords_23 = sub_23 + 13.0
        sub_23 = None
        long_23 = relative_coords_23.long()
        relative_coords_23 = None
        Rw_11 = l_self_modules_blocks_modules_11_modules_attn_parameters_rel_pos_w_[
            long_23
        ]
        l_self_modules_blocks_modules_11_modules_attn_parameters_rel_pos_w_ = (
            long_23
        ) = None
        r_q_11 = q_11.reshape(16, 14, 14, 80)
        rel_h_11 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_11, Rh_11)
        Rh_11 = None
        rel_w_11 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_11, Rw_11)
        r_q_11 = Rw_11 = None
        getitem_144 = rel_h_11[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_11 = None
        getitem_145 = rel_w_11[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_11 = None
        attn_bias_22 = getitem_144 + getitem_145
        getitem_144 = getitem_145 = None
        attn_bias_23 = attn_bias_22.reshape(-1, 196, 196)
        attn_bias_22 = None
        x_234 = torch._C._nn.scaled_dot_product_attention(
            q_11, k_11, v_11, attn_mask=attn_bias_23, dropout_p=0.0
        )
        q_11 = k_11 = v_11 = attn_bias_23 = None
        view_76 = x_234.view(1, 16, 196, -1)
        x_234 = None
        transpose_11 = view_76.transpose(1, 2)
        view_76 = None
        x_235 = transpose_11.reshape(1, 196, -1)
        transpose_11 = None
        x_236 = torch._C._nn.linear(
            x_235,
            l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_,
        )
        x_235 = l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_237 = torch.nn.functional.dropout(x_236, 0.0, False, False)
        x_236 = None
        x_238 = x_237.view(1, 14, 14, -1)
        x_237 = None
        x_239 = x_238.view(1, 1, 1, 14, 14, -1)
        x_238 = None
        permute_40 = x_239.permute(0, 1, 3, 2, 4, 5)
        x_239 = None
        contiguous_31 = permute_40.contiguous()
        permute_40 = None
        x_240 = contiguous_31.view(1, 14, 14, -1)
        contiguous_31 = None
        getitem_146 = x_240[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_240 = None
        x_241 = getitem_146.contiguous()
        getitem_146 = None
        x_242 = x_229 + x_241
        x_229 = x_241 = None
        x_243 = x_242.reshape(1, 196, -1)
        x_242 = None
        layer_norm_23 = torch.nn.functional.layer_norm(
            x_243,
            (1280,),
            l_self_modules_blocks_modules_11_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_11_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_11_modules_norm2_parameters_bias_
        ) = None
        x_244 = torch._C._nn.linear(
            layer_norm_23,
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_23 = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_245 = torch._C._nn.gelu(x_244, approximate="none")
        x_244 = None
        x_246 = torch.nn.functional.dropout(x_245, 0.0, False, False)
        x_245 = None
        x_247 = torch._C._nn.linear(
            x_246,
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_246 = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_248 = torch.nn.functional.dropout(x_247, 0.0, False, False)
        x_247 = None
        x_249 = x_243 + x_248
        x_243 = x_248 = None
        x_250 = x_249.reshape(1, 14, 14, -1)
        x_249 = None
        x_251 = torch.nn.functional.layer_norm(
            x_250,
            (1280,),
            l_self_modules_blocks_modules_12_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_12_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_12_modules_norm1_parameters_bias_
        ) = None
        x_252 = torch._C._nn.pad(x_251, (0, 0, 0, 0, 0, 0), "constant", None)
        x_251 = None
        x_253 = x_252.view(1, 1, 14, 1, 14, 1280)
        x_252 = None
        permute_41 = x_253.permute(0, 1, 3, 2, 4, 5)
        x_253 = None
        contiguous_33 = permute_41.contiguous()
        permute_41 = None
        windows_11 = contiguous_33.view(-1, 14, 14, 1280)
        contiguous_33 = None
        x_254 = windows_11.reshape(1, 196, -1)
        windows_11 = None
        linear_48 = torch._C._nn.linear(
            x_254,
            l_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_bias_,
        )
        x_254 = (
            l_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_82 = linear_48.view(1, 196, 3, 16, -1)
        linear_48 = None
        qkv_12 = view_82.permute(2, 0, 3, 1, 4)
        view_82 = None
        reshape_90 = qkv_12.reshape(3, 16, 196, -1)
        qkv_12 = None
        unbind_12 = reshape_90.unbind(0)
        reshape_90 = None
        q_12 = unbind_12[0]
        k_12 = unbind_12[1]
        v_12 = unbind_12[2]
        unbind_12 = None
        arange_48 = torch.arange(14)
        getitem_150 = arange_48[(slice(None, None, None), None)]
        arange_48 = None
        q_coords_24 = getitem_150 * 1.0
        getitem_150 = None
        arange_49 = torch.arange(14)
        getitem_151 = arange_49[(None, slice(None, None, None))]
        arange_49 = None
        k_coords_24 = getitem_151 * 1.0
        getitem_151 = None
        sub_24 = q_coords_24 - k_coords_24
        q_coords_24 = k_coords_24 = None
        relative_coords_24 = sub_24 + 13.0
        sub_24 = None
        long_24 = relative_coords_24.long()
        relative_coords_24 = None
        Rh_12 = l_self_modules_blocks_modules_12_modules_attn_parameters_rel_pos_h_[
            long_24
        ]
        l_self_modules_blocks_modules_12_modules_attn_parameters_rel_pos_h_ = (
            long_24
        ) = None
        arange_50 = torch.arange(14)
        getitem_153 = arange_50[(slice(None, None, None), None)]
        arange_50 = None
        q_coords_25 = getitem_153 * 1.0
        getitem_153 = None
        arange_51 = torch.arange(14)
        getitem_154 = arange_51[(None, slice(None, None, None))]
        arange_51 = None
        k_coords_25 = getitem_154 * 1.0
        getitem_154 = None
        sub_25 = q_coords_25 - k_coords_25
        q_coords_25 = k_coords_25 = None
        relative_coords_25 = sub_25 + 13.0
        sub_25 = None
        long_25 = relative_coords_25.long()
        relative_coords_25 = None
        Rw_12 = l_self_modules_blocks_modules_12_modules_attn_parameters_rel_pos_w_[
            long_25
        ]
        l_self_modules_blocks_modules_12_modules_attn_parameters_rel_pos_w_ = (
            long_25
        ) = None
        r_q_12 = q_12.reshape(16, 14, 14, 80)
        rel_h_12 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_12, Rh_12)
        Rh_12 = None
        rel_w_12 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_12, Rw_12)
        r_q_12 = Rw_12 = None
        getitem_156 = rel_h_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_12 = None
        getitem_157 = rel_w_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_12 = None
        attn_bias_24 = getitem_156 + getitem_157
        getitem_156 = getitem_157 = None
        attn_bias_25 = attn_bias_24.reshape(-1, 196, 196)
        attn_bias_24 = None
        x_255 = torch._C._nn.scaled_dot_product_attention(
            q_12, k_12, v_12, attn_mask=attn_bias_25, dropout_p=0.0
        )
        q_12 = k_12 = v_12 = attn_bias_25 = None
        view_83 = x_255.view(1, 16, 196, -1)
        x_255 = None
        transpose_12 = view_83.transpose(1, 2)
        view_83 = None
        x_256 = transpose_12.reshape(1, 196, -1)
        transpose_12 = None
        x_257 = torch._C._nn.linear(
            x_256,
            l_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_,
        )
        x_256 = l_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_258 = torch.nn.functional.dropout(x_257, 0.0, False, False)
        x_257 = None
        x_259 = x_258.view(1, 14, 14, -1)
        x_258 = None
        x_260 = x_259.view(1, 1, 1, 14, 14, -1)
        x_259 = None
        permute_43 = x_260.permute(0, 1, 3, 2, 4, 5)
        x_260 = None
        contiguous_34 = permute_43.contiguous()
        permute_43 = None
        x_261 = contiguous_34.view(1, 14, 14, -1)
        contiguous_34 = None
        getitem_158 = x_261[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_261 = None
        x_262 = getitem_158.contiguous()
        getitem_158 = None
        x_263 = x_250 + x_262
        x_250 = x_262 = None
        x_264 = x_263.reshape(1, 196, -1)
        x_263 = None
        layer_norm_25 = torch.nn.functional.layer_norm(
            x_264,
            (1280,),
            l_self_modules_blocks_modules_12_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_12_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_12_modules_norm2_parameters_bias_
        ) = None
        x_265 = torch._C._nn.linear(
            layer_norm_25,
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_25 = (
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_266 = torch._C._nn.gelu(x_265, approximate="none")
        x_265 = None
        x_267 = torch.nn.functional.dropout(x_266, 0.0, False, False)
        x_266 = None
        x_268 = torch._C._nn.linear(
            x_267,
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_267 = (
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_269 = torch.nn.functional.dropout(x_268, 0.0, False, False)
        x_268 = None
        x_270 = x_264 + x_269
        x_264 = x_269 = None
        x_271 = x_270.reshape(1, 14, 14, -1)
        x_270 = None
        x_272 = torch.nn.functional.layer_norm(
            x_271,
            (1280,),
            l_self_modules_blocks_modules_13_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_13_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_13_modules_norm1_parameters_bias_
        ) = None
        x_273 = torch._C._nn.pad(x_272, (0, 0, 0, 0, 0, 0), "constant", None)
        x_272 = None
        x_274 = x_273.view(1, 1, 14, 1, 14, 1280)
        x_273 = None
        permute_44 = x_274.permute(0, 1, 3, 2, 4, 5)
        x_274 = None
        contiguous_36 = permute_44.contiguous()
        permute_44 = None
        windows_12 = contiguous_36.view(-1, 14, 14, 1280)
        contiguous_36 = None
        x_275 = windows_12.reshape(1, 196, -1)
        windows_12 = None
        linear_52 = torch._C._nn.linear(
            x_275,
            l_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_bias_,
        )
        x_275 = (
            l_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_89 = linear_52.view(1, 196, 3, 16, -1)
        linear_52 = None
        qkv_13 = view_89.permute(2, 0, 3, 1, 4)
        view_89 = None
        reshape_97 = qkv_13.reshape(3, 16, 196, -1)
        qkv_13 = None
        unbind_13 = reshape_97.unbind(0)
        reshape_97 = None
        q_13 = unbind_13[0]
        k_13 = unbind_13[1]
        v_13 = unbind_13[2]
        unbind_13 = None
        arange_52 = torch.arange(14)
        getitem_162 = arange_52[(slice(None, None, None), None)]
        arange_52 = None
        q_coords_26 = getitem_162 * 1.0
        getitem_162 = None
        arange_53 = torch.arange(14)
        getitem_163 = arange_53[(None, slice(None, None, None))]
        arange_53 = None
        k_coords_26 = getitem_163 * 1.0
        getitem_163 = None
        sub_26 = q_coords_26 - k_coords_26
        q_coords_26 = k_coords_26 = None
        relative_coords_26 = sub_26 + 13.0
        sub_26 = None
        long_26 = relative_coords_26.long()
        relative_coords_26 = None
        Rh_13 = l_self_modules_blocks_modules_13_modules_attn_parameters_rel_pos_h_[
            long_26
        ]
        l_self_modules_blocks_modules_13_modules_attn_parameters_rel_pos_h_ = (
            long_26
        ) = None
        arange_54 = torch.arange(14)
        getitem_165 = arange_54[(slice(None, None, None), None)]
        arange_54 = None
        q_coords_27 = getitem_165 * 1.0
        getitem_165 = None
        arange_55 = torch.arange(14)
        getitem_166 = arange_55[(None, slice(None, None, None))]
        arange_55 = None
        k_coords_27 = getitem_166 * 1.0
        getitem_166 = None
        sub_27 = q_coords_27 - k_coords_27
        q_coords_27 = k_coords_27 = None
        relative_coords_27 = sub_27 + 13.0
        sub_27 = None
        long_27 = relative_coords_27.long()
        relative_coords_27 = None
        Rw_13 = l_self_modules_blocks_modules_13_modules_attn_parameters_rel_pos_w_[
            long_27
        ]
        l_self_modules_blocks_modules_13_modules_attn_parameters_rel_pos_w_ = (
            long_27
        ) = None
        r_q_13 = q_13.reshape(16, 14, 14, 80)
        rel_h_13 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_13, Rh_13)
        Rh_13 = None
        rel_w_13 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_13, Rw_13)
        r_q_13 = Rw_13 = None
        getitem_168 = rel_h_13[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_13 = None
        getitem_169 = rel_w_13[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_13 = None
        attn_bias_26 = getitem_168 + getitem_169
        getitem_168 = getitem_169 = None
        attn_bias_27 = attn_bias_26.reshape(-1, 196, 196)
        attn_bias_26 = None
        x_276 = torch._C._nn.scaled_dot_product_attention(
            q_13, k_13, v_13, attn_mask=attn_bias_27, dropout_p=0.0
        )
        q_13 = k_13 = v_13 = attn_bias_27 = None
        view_90 = x_276.view(1, 16, 196, -1)
        x_276 = None
        transpose_13 = view_90.transpose(1, 2)
        view_90 = None
        x_277 = transpose_13.reshape(1, 196, -1)
        transpose_13 = None
        x_278 = torch._C._nn.linear(
            x_277,
            l_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_,
        )
        x_277 = l_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_279 = torch.nn.functional.dropout(x_278, 0.0, False, False)
        x_278 = None
        x_280 = x_279.view(1, 14, 14, -1)
        x_279 = None
        x_281 = x_280.view(1, 1, 1, 14, 14, -1)
        x_280 = None
        permute_46 = x_281.permute(0, 1, 3, 2, 4, 5)
        x_281 = None
        contiguous_37 = permute_46.contiguous()
        permute_46 = None
        x_282 = contiguous_37.view(1, 14, 14, -1)
        contiguous_37 = None
        getitem_170 = x_282[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_282 = None
        x_283 = getitem_170.contiguous()
        getitem_170 = None
        x_284 = x_271 + x_283
        x_271 = x_283 = None
        x_285 = x_284.reshape(1, 196, -1)
        x_284 = None
        layer_norm_27 = torch.nn.functional.layer_norm(
            x_285,
            (1280,),
            l_self_modules_blocks_modules_13_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_13_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_13_modules_norm2_parameters_bias_
        ) = None
        x_286 = torch._C._nn.linear(
            layer_norm_27,
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_27 = (
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_287 = torch._C._nn.gelu(x_286, approximate="none")
        x_286 = None
        x_288 = torch.nn.functional.dropout(x_287, 0.0, False, False)
        x_287 = None
        x_289 = torch._C._nn.linear(
            x_288,
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_288 = (
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_290 = torch.nn.functional.dropout(x_289, 0.0, False, False)
        x_289 = None
        x_291 = x_285 + x_290
        x_285 = x_290 = None
        x_292 = x_291.reshape(1, 14, 14, -1)
        x_291 = None
        x_293 = torch.nn.functional.layer_norm(
            x_292,
            (1280,),
            l_self_modules_blocks_modules_14_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_14_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_14_modules_norm1_parameters_bias_
        ) = None
        x_294 = torch._C._nn.pad(x_293, (0, 0, 0, 0, 0, 0), "constant", None)
        x_293 = None
        x_295 = x_294.view(1, 1, 14, 1, 14, 1280)
        x_294 = None
        permute_47 = x_295.permute(0, 1, 3, 2, 4, 5)
        x_295 = None
        contiguous_39 = permute_47.contiguous()
        permute_47 = None
        windows_13 = contiguous_39.view(-1, 14, 14, 1280)
        contiguous_39 = None
        x_296 = windows_13.reshape(1, 196, -1)
        windows_13 = None
        linear_56 = torch._C._nn.linear(
            x_296,
            l_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_bias_,
        )
        x_296 = (
            l_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_96 = linear_56.view(1, 196, 3, 16, -1)
        linear_56 = None
        qkv_14 = view_96.permute(2, 0, 3, 1, 4)
        view_96 = None
        reshape_104 = qkv_14.reshape(3, 16, 196, -1)
        qkv_14 = None
        unbind_14 = reshape_104.unbind(0)
        reshape_104 = None
        q_14 = unbind_14[0]
        k_14 = unbind_14[1]
        v_14 = unbind_14[2]
        unbind_14 = None
        arange_56 = torch.arange(14)
        getitem_174 = arange_56[(slice(None, None, None), None)]
        arange_56 = None
        q_coords_28 = getitem_174 * 1.0
        getitem_174 = None
        arange_57 = torch.arange(14)
        getitem_175 = arange_57[(None, slice(None, None, None))]
        arange_57 = None
        k_coords_28 = getitem_175 * 1.0
        getitem_175 = None
        sub_28 = q_coords_28 - k_coords_28
        q_coords_28 = k_coords_28 = None
        relative_coords_28 = sub_28 + 13.0
        sub_28 = None
        long_28 = relative_coords_28.long()
        relative_coords_28 = None
        Rh_14 = l_self_modules_blocks_modules_14_modules_attn_parameters_rel_pos_h_[
            long_28
        ]
        l_self_modules_blocks_modules_14_modules_attn_parameters_rel_pos_h_ = (
            long_28
        ) = None
        arange_58 = torch.arange(14)
        getitem_177 = arange_58[(slice(None, None, None), None)]
        arange_58 = None
        q_coords_29 = getitem_177 * 1.0
        getitem_177 = None
        arange_59 = torch.arange(14)
        getitem_178 = arange_59[(None, slice(None, None, None))]
        arange_59 = None
        k_coords_29 = getitem_178 * 1.0
        getitem_178 = None
        sub_29 = q_coords_29 - k_coords_29
        q_coords_29 = k_coords_29 = None
        relative_coords_29 = sub_29 + 13.0
        sub_29 = None
        long_29 = relative_coords_29.long()
        relative_coords_29 = None
        Rw_14 = l_self_modules_blocks_modules_14_modules_attn_parameters_rel_pos_w_[
            long_29
        ]
        l_self_modules_blocks_modules_14_modules_attn_parameters_rel_pos_w_ = (
            long_29
        ) = None
        r_q_14 = q_14.reshape(16, 14, 14, 80)
        rel_h_14 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_14, Rh_14)
        Rh_14 = None
        rel_w_14 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_14, Rw_14)
        r_q_14 = Rw_14 = None
        getitem_180 = rel_h_14[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_14 = None
        getitem_181 = rel_w_14[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_14 = None
        attn_bias_28 = getitem_180 + getitem_181
        getitem_180 = getitem_181 = None
        attn_bias_29 = attn_bias_28.reshape(-1, 196, 196)
        attn_bias_28 = None
        x_297 = torch._C._nn.scaled_dot_product_attention(
            q_14, k_14, v_14, attn_mask=attn_bias_29, dropout_p=0.0
        )
        q_14 = k_14 = v_14 = attn_bias_29 = None
        view_97 = x_297.view(1, 16, 196, -1)
        x_297 = None
        transpose_14 = view_97.transpose(1, 2)
        view_97 = None
        x_298 = transpose_14.reshape(1, 196, -1)
        transpose_14 = None
        x_299 = torch._C._nn.linear(
            x_298,
            l_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_bias_,
        )
        x_298 = l_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_300 = torch.nn.functional.dropout(x_299, 0.0, False, False)
        x_299 = None
        x_301 = x_300.view(1, 14, 14, -1)
        x_300 = None
        x_302 = x_301.view(1, 1, 1, 14, 14, -1)
        x_301 = None
        permute_49 = x_302.permute(0, 1, 3, 2, 4, 5)
        x_302 = None
        contiguous_40 = permute_49.contiguous()
        permute_49 = None
        x_303 = contiguous_40.view(1, 14, 14, -1)
        contiguous_40 = None
        getitem_182 = x_303[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_303 = None
        x_304 = getitem_182.contiguous()
        getitem_182 = None
        x_305 = x_292 + x_304
        x_292 = x_304 = None
        x_306 = x_305.reshape(1, 196, -1)
        x_305 = None
        layer_norm_29 = torch.nn.functional.layer_norm(
            x_306,
            (1280,),
            l_self_modules_blocks_modules_14_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_14_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_14_modules_norm2_parameters_bias_
        ) = None
        x_307 = torch._C._nn.linear(
            layer_norm_29,
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_29 = (
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_308 = torch._C._nn.gelu(x_307, approximate="none")
        x_307 = None
        x_309 = torch.nn.functional.dropout(x_308, 0.0, False, False)
        x_308 = None
        x_310 = torch._C._nn.linear(
            x_309,
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_309 = (
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_311 = torch.nn.functional.dropout(x_310, 0.0, False, False)
        x_310 = None
        x_312 = x_306 + x_311
        x_306 = x_311 = None
        x_313 = x_312.reshape(1, 14, 14, -1)
        x_312 = None
        x_314 = torch.nn.functional.layer_norm(
            x_313,
            (1280,),
            l_self_modules_blocks_modules_15_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_15_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_15_modules_norm1_parameters_bias_
        ) = None
        x_315 = x_314.reshape(1, 196, -1)
        x_314 = None
        linear_60 = torch._C._nn.linear(
            x_315,
            l_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_bias_,
        )
        x_315 = (
            l_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_101 = linear_60.view(1, 196, 3, 16, -1)
        linear_60 = None
        qkv_15 = view_101.permute(2, 0, 3, 1, 4)
        view_101 = None
        reshape_111 = qkv_15.reshape(3, 16, 196, -1)
        qkv_15 = None
        unbind_15 = reshape_111.unbind(0)
        reshape_111 = None
        q_15 = unbind_15[0]
        k_15 = unbind_15[1]
        v_15 = unbind_15[2]
        unbind_15 = None
        reshape_112 = (
            l_self_modules_blocks_modules_15_modules_attn_parameters_rel_pos_h_.reshape(
                1, 127, -1
            )
        )
        l_self_modules_blocks_modules_15_modules_attn_parameters_rel_pos_h_ = None
        permute_51 = reshape_112.permute(0, 2, 1)
        reshape_112 = None
        rel_pos_resized_4 = torch.nn.functional.interpolate(
            permute_51, size=27, mode="linear"
        )
        permute_51 = None
        reshape_113 = rel_pos_resized_4.reshape(-1, 27)
        rel_pos_resized_4 = None
        rel_pos_resized_5 = reshape_113.permute(1, 0)
        reshape_113 = None
        arange_60 = torch.arange(14)
        getitem_186 = arange_60[(slice(None, None, None), None)]
        arange_60 = None
        q_coords_30 = getitem_186 * 1.0
        getitem_186 = None
        arange_61 = torch.arange(14)
        getitem_187 = arange_61[(None, slice(None, None, None))]
        arange_61 = None
        k_coords_30 = getitem_187 * 1.0
        getitem_187 = None
        sub_30 = q_coords_30 - k_coords_30
        q_coords_30 = k_coords_30 = None
        relative_coords_30 = sub_30 + 13.0
        sub_30 = None
        long_30 = relative_coords_30.long()
        relative_coords_30 = None
        Rh_15 = rel_pos_resized_5[long_30]
        rel_pos_resized_5 = long_30 = None
        reshape_114 = (
            l_self_modules_blocks_modules_15_modules_attn_parameters_rel_pos_w_.reshape(
                1, 127, -1
            )
        )
        l_self_modules_blocks_modules_15_modules_attn_parameters_rel_pos_w_ = None
        permute_53 = reshape_114.permute(0, 2, 1)
        reshape_114 = None
        rel_pos_resized_6 = torch.nn.functional.interpolate(
            permute_53, size=27, mode="linear"
        )
        permute_53 = None
        reshape_115 = rel_pos_resized_6.reshape(-1, 27)
        rel_pos_resized_6 = None
        rel_pos_resized_7 = reshape_115.permute(1, 0)
        reshape_115 = None
        arange_62 = torch.arange(14)
        getitem_189 = arange_62[(slice(None, None, None), None)]
        arange_62 = None
        q_coords_31 = getitem_189 * 1.0
        getitem_189 = None
        arange_63 = torch.arange(14)
        getitem_190 = arange_63[(None, slice(None, None, None))]
        arange_63 = None
        k_coords_31 = getitem_190 * 1.0
        getitem_190 = None
        sub_31 = q_coords_31 - k_coords_31
        q_coords_31 = k_coords_31 = None
        relative_coords_31 = sub_31 + 13.0
        sub_31 = None
        long_31 = relative_coords_31.long()
        relative_coords_31 = None
        Rw_15 = rel_pos_resized_7[long_31]
        rel_pos_resized_7 = long_31 = None
        r_q_15 = q_15.reshape(16, 14, 14, 80)
        rel_h_15 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_15, Rh_15)
        Rh_15 = None
        rel_w_15 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_15, Rw_15)
        r_q_15 = Rw_15 = None
        getitem_192 = rel_h_15[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_15 = None
        getitem_193 = rel_w_15[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_15 = None
        attn_bias_30 = getitem_192 + getitem_193
        getitem_192 = getitem_193 = None
        attn_bias_31 = attn_bias_30.reshape(-1, 196, 196)
        attn_bias_30 = None
        x_316 = torch._C._nn.scaled_dot_product_attention(
            q_15, k_15, v_15, attn_mask=attn_bias_31, dropout_p=0.0
        )
        q_15 = k_15 = v_15 = attn_bias_31 = None
        view_102 = x_316.view(1, 16, 196, -1)
        x_316 = None
        transpose_15 = view_102.transpose(1, 2)
        view_102 = None
        x_317 = transpose_15.reshape(1, 196, -1)
        transpose_15 = None
        x_318 = torch._C._nn.linear(
            x_317,
            l_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_bias_,
        )
        x_317 = l_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_319 = torch.nn.functional.dropout(x_318, 0.0, False, False)
        x_318 = None
        x_320 = x_319.view(1, 14, 14, -1)
        x_319 = None
        x_321 = x_313 + x_320
        x_313 = x_320 = None
        x_322 = x_321.reshape(1, 196, -1)
        x_321 = None
        layer_norm_31 = torch.nn.functional.layer_norm(
            x_322,
            (1280,),
            l_self_modules_blocks_modules_15_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_15_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_15_modules_norm2_parameters_bias_
        ) = None
        x_323 = torch._C._nn.linear(
            layer_norm_31,
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_31 = (
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_324 = torch._C._nn.gelu(x_323, approximate="none")
        x_323 = None
        x_325 = torch.nn.functional.dropout(x_324, 0.0, False, False)
        x_324 = None
        x_326 = torch._C._nn.linear(
            x_325,
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_325 = (
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_327 = torch.nn.functional.dropout(x_326, 0.0, False, False)
        x_326 = None
        x_328 = x_322 + x_327
        x_322 = x_327 = None
        x_329 = x_328.reshape(1, 14, 14, -1)
        x_328 = None
        x_330 = torch.nn.functional.layer_norm(
            x_329,
            (1280,),
            l_self_modules_blocks_modules_16_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_16_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_16_modules_norm1_parameters_bias_
        ) = None
        x_331 = torch._C._nn.pad(x_330, (0, 0, 0, 0, 0, 0), "constant", None)
        x_330 = None
        x_332 = x_331.view(1, 1, 14, 1, 14, 1280)
        x_331 = None
        permute_55 = x_332.permute(0, 1, 3, 2, 4, 5)
        x_332 = None
        contiguous_42 = permute_55.contiguous()
        permute_55 = None
        windows_14 = contiguous_42.view(-1, 14, 14, 1280)
        contiguous_42 = None
        x_333 = windows_14.reshape(1, 196, -1)
        windows_14 = None
        linear_64 = torch._C._nn.linear(
            x_333,
            l_self_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_bias_,
        )
        x_333 = (
            l_self_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_106 = linear_64.view(1, 196, 3, 16, -1)
        linear_64 = None
        qkv_16 = view_106.permute(2, 0, 3, 1, 4)
        view_106 = None
        reshape_122 = qkv_16.reshape(3, 16, 196, -1)
        qkv_16 = None
        unbind_16 = reshape_122.unbind(0)
        reshape_122 = None
        q_16 = unbind_16[0]
        k_16 = unbind_16[1]
        v_16 = unbind_16[2]
        unbind_16 = None
        arange_64 = torch.arange(14)
        getitem_197 = arange_64[(slice(None, None, None), None)]
        arange_64 = None
        q_coords_32 = getitem_197 * 1.0
        getitem_197 = None
        arange_65 = torch.arange(14)
        getitem_198 = arange_65[(None, slice(None, None, None))]
        arange_65 = None
        k_coords_32 = getitem_198 * 1.0
        getitem_198 = None
        sub_32 = q_coords_32 - k_coords_32
        q_coords_32 = k_coords_32 = None
        relative_coords_32 = sub_32 + 13.0
        sub_32 = None
        long_32 = relative_coords_32.long()
        relative_coords_32 = None
        Rh_16 = l_self_modules_blocks_modules_16_modules_attn_parameters_rel_pos_h_[
            long_32
        ]
        l_self_modules_blocks_modules_16_modules_attn_parameters_rel_pos_h_ = (
            long_32
        ) = None
        arange_66 = torch.arange(14)
        getitem_200 = arange_66[(slice(None, None, None), None)]
        arange_66 = None
        q_coords_33 = getitem_200 * 1.0
        getitem_200 = None
        arange_67 = torch.arange(14)
        getitem_201 = arange_67[(None, slice(None, None, None))]
        arange_67 = None
        k_coords_33 = getitem_201 * 1.0
        getitem_201 = None
        sub_33 = q_coords_33 - k_coords_33
        q_coords_33 = k_coords_33 = None
        relative_coords_33 = sub_33 + 13.0
        sub_33 = None
        long_33 = relative_coords_33.long()
        relative_coords_33 = None
        Rw_16 = l_self_modules_blocks_modules_16_modules_attn_parameters_rel_pos_w_[
            long_33
        ]
        l_self_modules_blocks_modules_16_modules_attn_parameters_rel_pos_w_ = (
            long_33
        ) = None
        r_q_16 = q_16.reshape(16, 14, 14, 80)
        rel_h_16 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_16, Rh_16)
        Rh_16 = None
        rel_w_16 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_16, Rw_16)
        r_q_16 = Rw_16 = None
        getitem_203 = rel_h_16[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_16 = None
        getitem_204 = rel_w_16[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_16 = None
        attn_bias_32 = getitem_203 + getitem_204
        getitem_203 = getitem_204 = None
        attn_bias_33 = attn_bias_32.reshape(-1, 196, 196)
        attn_bias_32 = None
        x_334 = torch._C._nn.scaled_dot_product_attention(
            q_16, k_16, v_16, attn_mask=attn_bias_33, dropout_p=0.0
        )
        q_16 = k_16 = v_16 = attn_bias_33 = None
        view_107 = x_334.view(1, 16, 196, -1)
        x_334 = None
        transpose_16 = view_107.transpose(1, 2)
        view_107 = None
        x_335 = transpose_16.reshape(1, 196, -1)
        transpose_16 = None
        x_336 = torch._C._nn.linear(
            x_335,
            l_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_bias_,
        )
        x_335 = l_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_337 = torch.nn.functional.dropout(x_336, 0.0, False, False)
        x_336 = None
        x_338 = x_337.view(1, 14, 14, -1)
        x_337 = None
        x_339 = x_338.view(1, 1, 1, 14, 14, -1)
        x_338 = None
        permute_57 = x_339.permute(0, 1, 3, 2, 4, 5)
        x_339 = None
        contiguous_43 = permute_57.contiguous()
        permute_57 = None
        x_340 = contiguous_43.view(1, 14, 14, -1)
        contiguous_43 = None
        getitem_205 = x_340[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_340 = None
        x_341 = getitem_205.contiguous()
        getitem_205 = None
        x_342 = x_329 + x_341
        x_329 = x_341 = None
        x_343 = x_342.reshape(1, 196, -1)
        x_342 = None
        layer_norm_33 = torch.nn.functional.layer_norm(
            x_343,
            (1280,),
            l_self_modules_blocks_modules_16_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_16_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_16_modules_norm2_parameters_bias_
        ) = None
        x_344 = torch._C._nn.linear(
            layer_norm_33,
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_33 = (
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_345 = torch._C._nn.gelu(x_344, approximate="none")
        x_344 = None
        x_346 = torch.nn.functional.dropout(x_345, 0.0, False, False)
        x_345 = None
        x_347 = torch._C._nn.linear(
            x_346,
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_346 = (
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_348 = torch.nn.functional.dropout(x_347, 0.0, False, False)
        x_347 = None
        x_349 = x_343 + x_348
        x_343 = x_348 = None
        x_350 = x_349.reshape(1, 14, 14, -1)
        x_349 = None
        x_351 = torch.nn.functional.layer_norm(
            x_350,
            (1280,),
            l_self_modules_blocks_modules_17_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_17_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_17_modules_norm1_parameters_bias_
        ) = None
        x_352 = torch._C._nn.pad(x_351, (0, 0, 0, 0, 0, 0), "constant", None)
        x_351 = None
        x_353 = x_352.view(1, 1, 14, 1, 14, 1280)
        x_352 = None
        permute_58 = x_353.permute(0, 1, 3, 2, 4, 5)
        x_353 = None
        contiguous_45 = permute_58.contiguous()
        permute_58 = None
        windows_15 = contiguous_45.view(-1, 14, 14, 1280)
        contiguous_45 = None
        x_354 = windows_15.reshape(1, 196, -1)
        windows_15 = None
        linear_68 = torch._C._nn.linear(
            x_354,
            l_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_bias_,
        )
        x_354 = (
            l_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_113 = linear_68.view(1, 196, 3, 16, -1)
        linear_68 = None
        qkv_17 = view_113.permute(2, 0, 3, 1, 4)
        view_113 = None
        reshape_129 = qkv_17.reshape(3, 16, 196, -1)
        qkv_17 = None
        unbind_17 = reshape_129.unbind(0)
        reshape_129 = None
        q_17 = unbind_17[0]
        k_17 = unbind_17[1]
        v_17 = unbind_17[2]
        unbind_17 = None
        arange_68 = torch.arange(14)
        getitem_209 = arange_68[(slice(None, None, None), None)]
        arange_68 = None
        q_coords_34 = getitem_209 * 1.0
        getitem_209 = None
        arange_69 = torch.arange(14)
        getitem_210 = arange_69[(None, slice(None, None, None))]
        arange_69 = None
        k_coords_34 = getitem_210 * 1.0
        getitem_210 = None
        sub_34 = q_coords_34 - k_coords_34
        q_coords_34 = k_coords_34 = None
        relative_coords_34 = sub_34 + 13.0
        sub_34 = None
        long_34 = relative_coords_34.long()
        relative_coords_34 = None
        Rh_17 = l_self_modules_blocks_modules_17_modules_attn_parameters_rel_pos_h_[
            long_34
        ]
        l_self_modules_blocks_modules_17_modules_attn_parameters_rel_pos_h_ = (
            long_34
        ) = None
        arange_70 = torch.arange(14)
        getitem_212 = arange_70[(slice(None, None, None), None)]
        arange_70 = None
        q_coords_35 = getitem_212 * 1.0
        getitem_212 = None
        arange_71 = torch.arange(14)
        getitem_213 = arange_71[(None, slice(None, None, None))]
        arange_71 = None
        k_coords_35 = getitem_213 * 1.0
        getitem_213 = None
        sub_35 = q_coords_35 - k_coords_35
        q_coords_35 = k_coords_35 = None
        relative_coords_35 = sub_35 + 13.0
        sub_35 = None
        long_35 = relative_coords_35.long()
        relative_coords_35 = None
        Rw_17 = l_self_modules_blocks_modules_17_modules_attn_parameters_rel_pos_w_[
            long_35
        ]
        l_self_modules_blocks_modules_17_modules_attn_parameters_rel_pos_w_ = (
            long_35
        ) = None
        r_q_17 = q_17.reshape(16, 14, 14, 80)
        rel_h_17 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_17, Rh_17)
        Rh_17 = None
        rel_w_17 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_17, Rw_17)
        r_q_17 = Rw_17 = None
        getitem_215 = rel_h_17[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_17 = None
        getitem_216 = rel_w_17[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_17 = None
        attn_bias_34 = getitem_215 + getitem_216
        getitem_215 = getitem_216 = None
        attn_bias_35 = attn_bias_34.reshape(-1, 196, 196)
        attn_bias_34 = None
        x_355 = torch._C._nn.scaled_dot_product_attention(
            q_17, k_17, v_17, attn_mask=attn_bias_35, dropout_p=0.0
        )
        q_17 = k_17 = v_17 = attn_bias_35 = None
        view_114 = x_355.view(1, 16, 196, -1)
        x_355 = None
        transpose_17 = view_114.transpose(1, 2)
        view_114 = None
        x_356 = transpose_17.reshape(1, 196, -1)
        transpose_17 = None
        x_357 = torch._C._nn.linear(
            x_356,
            l_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_bias_,
        )
        x_356 = l_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_358 = torch.nn.functional.dropout(x_357, 0.0, False, False)
        x_357 = None
        x_359 = x_358.view(1, 14, 14, -1)
        x_358 = None
        x_360 = x_359.view(1, 1, 1, 14, 14, -1)
        x_359 = None
        permute_60 = x_360.permute(0, 1, 3, 2, 4, 5)
        x_360 = None
        contiguous_46 = permute_60.contiguous()
        permute_60 = None
        x_361 = contiguous_46.view(1, 14, 14, -1)
        contiguous_46 = None
        getitem_217 = x_361[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_361 = None
        x_362 = getitem_217.contiguous()
        getitem_217 = None
        x_363 = x_350 + x_362
        x_350 = x_362 = None
        x_364 = x_363.reshape(1, 196, -1)
        x_363 = None
        layer_norm_35 = torch.nn.functional.layer_norm(
            x_364,
            (1280,),
            l_self_modules_blocks_modules_17_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_17_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_17_modules_norm2_parameters_bias_
        ) = None
        x_365 = torch._C._nn.linear(
            layer_norm_35,
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_35 = (
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_366 = torch._C._nn.gelu(x_365, approximate="none")
        x_365 = None
        x_367 = torch.nn.functional.dropout(x_366, 0.0, False, False)
        x_366 = None
        x_368 = torch._C._nn.linear(
            x_367,
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_367 = (
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_369 = torch.nn.functional.dropout(x_368, 0.0, False, False)
        x_368 = None
        x_370 = x_364 + x_369
        x_364 = x_369 = None
        x_371 = x_370.reshape(1, 14, 14, -1)
        x_370 = None
        x_372 = torch.nn.functional.layer_norm(
            x_371,
            (1280,),
            l_self_modules_blocks_modules_18_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_18_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_18_modules_norm1_parameters_bias_
        ) = None
        x_373 = torch._C._nn.pad(x_372, (0, 0, 0, 0, 0, 0), "constant", None)
        x_372 = None
        x_374 = x_373.view(1, 1, 14, 1, 14, 1280)
        x_373 = None
        permute_61 = x_374.permute(0, 1, 3, 2, 4, 5)
        x_374 = None
        contiguous_48 = permute_61.contiguous()
        permute_61 = None
        windows_16 = contiguous_48.view(-1, 14, 14, 1280)
        contiguous_48 = None
        x_375 = windows_16.reshape(1, 196, -1)
        windows_16 = None
        linear_72 = torch._C._nn.linear(
            x_375,
            l_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_bias_,
        )
        x_375 = (
            l_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_120 = linear_72.view(1, 196, 3, 16, -1)
        linear_72 = None
        qkv_18 = view_120.permute(2, 0, 3, 1, 4)
        view_120 = None
        reshape_136 = qkv_18.reshape(3, 16, 196, -1)
        qkv_18 = None
        unbind_18 = reshape_136.unbind(0)
        reshape_136 = None
        q_18 = unbind_18[0]
        k_18 = unbind_18[1]
        v_18 = unbind_18[2]
        unbind_18 = None
        arange_72 = torch.arange(14)
        getitem_221 = arange_72[(slice(None, None, None), None)]
        arange_72 = None
        q_coords_36 = getitem_221 * 1.0
        getitem_221 = None
        arange_73 = torch.arange(14)
        getitem_222 = arange_73[(None, slice(None, None, None))]
        arange_73 = None
        k_coords_36 = getitem_222 * 1.0
        getitem_222 = None
        sub_36 = q_coords_36 - k_coords_36
        q_coords_36 = k_coords_36 = None
        relative_coords_36 = sub_36 + 13.0
        sub_36 = None
        long_36 = relative_coords_36.long()
        relative_coords_36 = None
        Rh_18 = l_self_modules_blocks_modules_18_modules_attn_parameters_rel_pos_h_[
            long_36
        ]
        l_self_modules_blocks_modules_18_modules_attn_parameters_rel_pos_h_ = (
            long_36
        ) = None
        arange_74 = torch.arange(14)
        getitem_224 = arange_74[(slice(None, None, None), None)]
        arange_74 = None
        q_coords_37 = getitem_224 * 1.0
        getitem_224 = None
        arange_75 = torch.arange(14)
        getitem_225 = arange_75[(None, slice(None, None, None))]
        arange_75 = None
        k_coords_37 = getitem_225 * 1.0
        getitem_225 = None
        sub_37 = q_coords_37 - k_coords_37
        q_coords_37 = k_coords_37 = None
        relative_coords_37 = sub_37 + 13.0
        sub_37 = None
        long_37 = relative_coords_37.long()
        relative_coords_37 = None
        Rw_18 = l_self_modules_blocks_modules_18_modules_attn_parameters_rel_pos_w_[
            long_37
        ]
        l_self_modules_blocks_modules_18_modules_attn_parameters_rel_pos_w_ = (
            long_37
        ) = None
        r_q_18 = q_18.reshape(16, 14, 14, 80)
        rel_h_18 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_18, Rh_18)
        Rh_18 = None
        rel_w_18 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_18, Rw_18)
        r_q_18 = Rw_18 = None
        getitem_227 = rel_h_18[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_18 = None
        getitem_228 = rel_w_18[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_18 = None
        attn_bias_36 = getitem_227 + getitem_228
        getitem_227 = getitem_228 = None
        attn_bias_37 = attn_bias_36.reshape(-1, 196, 196)
        attn_bias_36 = None
        x_376 = torch._C._nn.scaled_dot_product_attention(
            q_18, k_18, v_18, attn_mask=attn_bias_37, dropout_p=0.0
        )
        q_18 = k_18 = v_18 = attn_bias_37 = None
        view_121 = x_376.view(1, 16, 196, -1)
        x_376 = None
        transpose_18 = view_121.transpose(1, 2)
        view_121 = None
        x_377 = transpose_18.reshape(1, 196, -1)
        transpose_18 = None
        x_378 = torch._C._nn.linear(
            x_377,
            l_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_bias_,
        )
        x_377 = l_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_379 = torch.nn.functional.dropout(x_378, 0.0, False, False)
        x_378 = None
        x_380 = x_379.view(1, 14, 14, -1)
        x_379 = None
        x_381 = x_380.view(1, 1, 1, 14, 14, -1)
        x_380 = None
        permute_63 = x_381.permute(0, 1, 3, 2, 4, 5)
        x_381 = None
        contiguous_49 = permute_63.contiguous()
        permute_63 = None
        x_382 = contiguous_49.view(1, 14, 14, -1)
        contiguous_49 = None
        getitem_229 = x_382[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_382 = None
        x_383 = getitem_229.contiguous()
        getitem_229 = None
        x_384 = x_371 + x_383
        x_371 = x_383 = None
        x_385 = x_384.reshape(1, 196, -1)
        x_384 = None
        layer_norm_37 = torch.nn.functional.layer_norm(
            x_385,
            (1280,),
            l_self_modules_blocks_modules_18_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_18_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_18_modules_norm2_parameters_bias_
        ) = None
        x_386 = torch._C._nn.linear(
            layer_norm_37,
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_37 = (
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_387 = torch._C._nn.gelu(x_386, approximate="none")
        x_386 = None
        x_388 = torch.nn.functional.dropout(x_387, 0.0, False, False)
        x_387 = None
        x_389 = torch._C._nn.linear(
            x_388,
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_388 = (
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_390 = torch.nn.functional.dropout(x_389, 0.0, False, False)
        x_389 = None
        x_391 = x_385 + x_390
        x_385 = x_390 = None
        x_392 = x_391.reshape(1, 14, 14, -1)
        x_391 = None
        x_393 = torch.nn.functional.layer_norm(
            x_392,
            (1280,),
            l_self_modules_blocks_modules_19_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_19_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_19_modules_norm1_parameters_bias_
        ) = None
        x_394 = torch._C._nn.pad(x_393, (0, 0, 0, 0, 0, 0), "constant", None)
        x_393 = None
        x_395 = x_394.view(1, 1, 14, 1, 14, 1280)
        x_394 = None
        permute_64 = x_395.permute(0, 1, 3, 2, 4, 5)
        x_395 = None
        contiguous_51 = permute_64.contiguous()
        permute_64 = None
        windows_17 = contiguous_51.view(-1, 14, 14, 1280)
        contiguous_51 = None
        x_396 = windows_17.reshape(1, 196, -1)
        windows_17 = None
        linear_76 = torch._C._nn.linear(
            x_396,
            l_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_bias_,
        )
        x_396 = (
            l_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_127 = linear_76.view(1, 196, 3, 16, -1)
        linear_76 = None
        qkv_19 = view_127.permute(2, 0, 3, 1, 4)
        view_127 = None
        reshape_143 = qkv_19.reshape(3, 16, 196, -1)
        qkv_19 = None
        unbind_19 = reshape_143.unbind(0)
        reshape_143 = None
        q_19 = unbind_19[0]
        k_19 = unbind_19[1]
        v_19 = unbind_19[2]
        unbind_19 = None
        arange_76 = torch.arange(14)
        getitem_233 = arange_76[(slice(None, None, None), None)]
        arange_76 = None
        q_coords_38 = getitem_233 * 1.0
        getitem_233 = None
        arange_77 = torch.arange(14)
        getitem_234 = arange_77[(None, slice(None, None, None))]
        arange_77 = None
        k_coords_38 = getitem_234 * 1.0
        getitem_234 = None
        sub_38 = q_coords_38 - k_coords_38
        q_coords_38 = k_coords_38 = None
        relative_coords_38 = sub_38 + 13.0
        sub_38 = None
        long_38 = relative_coords_38.long()
        relative_coords_38 = None
        Rh_19 = l_self_modules_blocks_modules_19_modules_attn_parameters_rel_pos_h_[
            long_38
        ]
        l_self_modules_blocks_modules_19_modules_attn_parameters_rel_pos_h_ = (
            long_38
        ) = None
        arange_78 = torch.arange(14)
        getitem_236 = arange_78[(slice(None, None, None), None)]
        arange_78 = None
        q_coords_39 = getitem_236 * 1.0
        getitem_236 = None
        arange_79 = torch.arange(14)
        getitem_237 = arange_79[(None, slice(None, None, None))]
        arange_79 = None
        k_coords_39 = getitem_237 * 1.0
        getitem_237 = None
        sub_39 = q_coords_39 - k_coords_39
        q_coords_39 = k_coords_39 = None
        relative_coords_39 = sub_39 + 13.0
        sub_39 = None
        long_39 = relative_coords_39.long()
        relative_coords_39 = None
        Rw_19 = l_self_modules_blocks_modules_19_modules_attn_parameters_rel_pos_w_[
            long_39
        ]
        l_self_modules_blocks_modules_19_modules_attn_parameters_rel_pos_w_ = (
            long_39
        ) = None
        r_q_19 = q_19.reshape(16, 14, 14, 80)
        rel_h_19 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_19, Rh_19)
        Rh_19 = None
        rel_w_19 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_19, Rw_19)
        r_q_19 = Rw_19 = None
        getitem_239 = rel_h_19[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_19 = None
        getitem_240 = rel_w_19[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_19 = None
        attn_bias_38 = getitem_239 + getitem_240
        getitem_239 = getitem_240 = None
        attn_bias_39 = attn_bias_38.reshape(-1, 196, 196)
        attn_bias_38 = None
        x_397 = torch._C._nn.scaled_dot_product_attention(
            q_19, k_19, v_19, attn_mask=attn_bias_39, dropout_p=0.0
        )
        q_19 = k_19 = v_19 = attn_bias_39 = None
        view_128 = x_397.view(1, 16, 196, -1)
        x_397 = None
        transpose_19 = view_128.transpose(1, 2)
        view_128 = None
        x_398 = transpose_19.reshape(1, 196, -1)
        transpose_19 = None
        x_399 = torch._C._nn.linear(
            x_398,
            l_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_bias_,
        )
        x_398 = l_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_400 = torch.nn.functional.dropout(x_399, 0.0, False, False)
        x_399 = None
        x_401 = x_400.view(1, 14, 14, -1)
        x_400 = None
        x_402 = x_401.view(1, 1, 1, 14, 14, -1)
        x_401 = None
        permute_66 = x_402.permute(0, 1, 3, 2, 4, 5)
        x_402 = None
        contiguous_52 = permute_66.contiguous()
        permute_66 = None
        x_403 = contiguous_52.view(1, 14, 14, -1)
        contiguous_52 = None
        getitem_241 = x_403[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_403 = None
        x_404 = getitem_241.contiguous()
        getitem_241 = None
        x_405 = x_392 + x_404
        x_392 = x_404 = None
        x_406 = x_405.reshape(1, 196, -1)
        x_405 = None
        layer_norm_39 = torch.nn.functional.layer_norm(
            x_406,
            (1280,),
            l_self_modules_blocks_modules_19_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_19_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_19_modules_norm2_parameters_bias_
        ) = None
        x_407 = torch._C._nn.linear(
            layer_norm_39,
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_39 = (
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_408 = torch._C._nn.gelu(x_407, approximate="none")
        x_407 = None
        x_409 = torch.nn.functional.dropout(x_408, 0.0, False, False)
        x_408 = None
        x_410 = torch._C._nn.linear(
            x_409,
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_409 = (
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_411 = torch.nn.functional.dropout(x_410, 0.0, False, False)
        x_410 = None
        x_412 = x_406 + x_411
        x_406 = x_411 = None
        x_413 = x_412.reshape(1, 14, 14, -1)
        x_412 = None
        x_414 = torch.nn.functional.layer_norm(
            x_413,
            (1280,),
            l_self_modules_blocks_modules_20_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_20_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_20_modules_norm1_parameters_bias_
        ) = None
        x_415 = torch._C._nn.pad(x_414, (0, 0, 0, 0, 0, 0), "constant", None)
        x_414 = None
        x_416 = x_415.view(1, 1, 14, 1, 14, 1280)
        x_415 = None
        permute_67 = x_416.permute(0, 1, 3, 2, 4, 5)
        x_416 = None
        contiguous_54 = permute_67.contiguous()
        permute_67 = None
        windows_18 = contiguous_54.view(-1, 14, 14, 1280)
        contiguous_54 = None
        x_417 = windows_18.reshape(1, 196, -1)
        windows_18 = None
        linear_80 = torch._C._nn.linear(
            x_417,
            l_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_bias_,
        )
        x_417 = (
            l_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_134 = linear_80.view(1, 196, 3, 16, -1)
        linear_80 = None
        qkv_20 = view_134.permute(2, 0, 3, 1, 4)
        view_134 = None
        reshape_150 = qkv_20.reshape(3, 16, 196, -1)
        qkv_20 = None
        unbind_20 = reshape_150.unbind(0)
        reshape_150 = None
        q_20 = unbind_20[0]
        k_20 = unbind_20[1]
        v_20 = unbind_20[2]
        unbind_20 = None
        arange_80 = torch.arange(14)
        getitem_245 = arange_80[(slice(None, None, None), None)]
        arange_80 = None
        q_coords_40 = getitem_245 * 1.0
        getitem_245 = None
        arange_81 = torch.arange(14)
        getitem_246 = arange_81[(None, slice(None, None, None))]
        arange_81 = None
        k_coords_40 = getitem_246 * 1.0
        getitem_246 = None
        sub_40 = q_coords_40 - k_coords_40
        q_coords_40 = k_coords_40 = None
        relative_coords_40 = sub_40 + 13.0
        sub_40 = None
        long_40 = relative_coords_40.long()
        relative_coords_40 = None
        Rh_20 = l_self_modules_blocks_modules_20_modules_attn_parameters_rel_pos_h_[
            long_40
        ]
        l_self_modules_blocks_modules_20_modules_attn_parameters_rel_pos_h_ = (
            long_40
        ) = None
        arange_82 = torch.arange(14)
        getitem_248 = arange_82[(slice(None, None, None), None)]
        arange_82 = None
        q_coords_41 = getitem_248 * 1.0
        getitem_248 = None
        arange_83 = torch.arange(14)
        getitem_249 = arange_83[(None, slice(None, None, None))]
        arange_83 = None
        k_coords_41 = getitem_249 * 1.0
        getitem_249 = None
        sub_41 = q_coords_41 - k_coords_41
        q_coords_41 = k_coords_41 = None
        relative_coords_41 = sub_41 + 13.0
        sub_41 = None
        long_41 = relative_coords_41.long()
        relative_coords_41 = None
        Rw_20 = l_self_modules_blocks_modules_20_modules_attn_parameters_rel_pos_w_[
            long_41
        ]
        l_self_modules_blocks_modules_20_modules_attn_parameters_rel_pos_w_ = (
            long_41
        ) = None
        r_q_20 = q_20.reshape(16, 14, 14, 80)
        rel_h_20 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_20, Rh_20)
        Rh_20 = None
        rel_w_20 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_20, Rw_20)
        r_q_20 = Rw_20 = None
        getitem_251 = rel_h_20[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_20 = None
        getitem_252 = rel_w_20[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_20 = None
        attn_bias_40 = getitem_251 + getitem_252
        getitem_251 = getitem_252 = None
        attn_bias_41 = attn_bias_40.reshape(-1, 196, 196)
        attn_bias_40 = None
        x_418 = torch._C._nn.scaled_dot_product_attention(
            q_20, k_20, v_20, attn_mask=attn_bias_41, dropout_p=0.0
        )
        q_20 = k_20 = v_20 = attn_bias_41 = None
        view_135 = x_418.view(1, 16, 196, -1)
        x_418 = None
        transpose_20 = view_135.transpose(1, 2)
        view_135 = None
        x_419 = transpose_20.reshape(1, 196, -1)
        transpose_20 = None
        x_420 = torch._C._nn.linear(
            x_419,
            l_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_bias_,
        )
        x_419 = l_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_421 = torch.nn.functional.dropout(x_420, 0.0, False, False)
        x_420 = None
        x_422 = x_421.view(1, 14, 14, -1)
        x_421 = None
        x_423 = x_422.view(1, 1, 1, 14, 14, -1)
        x_422 = None
        permute_69 = x_423.permute(0, 1, 3, 2, 4, 5)
        x_423 = None
        contiguous_55 = permute_69.contiguous()
        permute_69 = None
        x_424 = contiguous_55.view(1, 14, 14, -1)
        contiguous_55 = None
        getitem_253 = x_424[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_424 = None
        x_425 = getitem_253.contiguous()
        getitem_253 = None
        x_426 = x_413 + x_425
        x_413 = x_425 = None
        x_427 = x_426.reshape(1, 196, -1)
        x_426 = None
        layer_norm_41 = torch.nn.functional.layer_norm(
            x_427,
            (1280,),
            l_self_modules_blocks_modules_20_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_20_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_20_modules_norm2_parameters_bias_
        ) = None
        x_428 = torch._C._nn.linear(
            layer_norm_41,
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_41 = (
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_429 = torch._C._nn.gelu(x_428, approximate="none")
        x_428 = None
        x_430 = torch.nn.functional.dropout(x_429, 0.0, False, False)
        x_429 = None
        x_431 = torch._C._nn.linear(
            x_430,
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_430 = (
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_432 = torch.nn.functional.dropout(x_431, 0.0, False, False)
        x_431 = None
        x_433 = x_427 + x_432
        x_427 = x_432 = None
        x_434 = x_433.reshape(1, 14, 14, -1)
        x_433 = None
        x_435 = torch.nn.functional.layer_norm(
            x_434,
            (1280,),
            l_self_modules_blocks_modules_21_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_21_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_21_modules_norm1_parameters_bias_
        ) = None
        x_436 = torch._C._nn.pad(x_435, (0, 0, 0, 0, 0, 0), "constant", None)
        x_435 = None
        x_437 = x_436.view(1, 1, 14, 1, 14, 1280)
        x_436 = None
        permute_70 = x_437.permute(0, 1, 3, 2, 4, 5)
        x_437 = None
        contiguous_57 = permute_70.contiguous()
        permute_70 = None
        windows_19 = contiguous_57.view(-1, 14, 14, 1280)
        contiguous_57 = None
        x_438 = windows_19.reshape(1, 196, -1)
        windows_19 = None
        linear_84 = torch._C._nn.linear(
            x_438,
            l_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_bias_,
        )
        x_438 = (
            l_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_141 = linear_84.view(1, 196, 3, 16, -1)
        linear_84 = None
        qkv_21 = view_141.permute(2, 0, 3, 1, 4)
        view_141 = None
        reshape_157 = qkv_21.reshape(3, 16, 196, -1)
        qkv_21 = None
        unbind_21 = reshape_157.unbind(0)
        reshape_157 = None
        q_21 = unbind_21[0]
        k_21 = unbind_21[1]
        v_21 = unbind_21[2]
        unbind_21 = None
        arange_84 = torch.arange(14)
        getitem_257 = arange_84[(slice(None, None, None), None)]
        arange_84 = None
        q_coords_42 = getitem_257 * 1.0
        getitem_257 = None
        arange_85 = torch.arange(14)
        getitem_258 = arange_85[(None, slice(None, None, None))]
        arange_85 = None
        k_coords_42 = getitem_258 * 1.0
        getitem_258 = None
        sub_42 = q_coords_42 - k_coords_42
        q_coords_42 = k_coords_42 = None
        relative_coords_42 = sub_42 + 13.0
        sub_42 = None
        long_42 = relative_coords_42.long()
        relative_coords_42 = None
        Rh_21 = l_self_modules_blocks_modules_21_modules_attn_parameters_rel_pos_h_[
            long_42
        ]
        l_self_modules_blocks_modules_21_modules_attn_parameters_rel_pos_h_ = (
            long_42
        ) = None
        arange_86 = torch.arange(14)
        getitem_260 = arange_86[(slice(None, None, None), None)]
        arange_86 = None
        q_coords_43 = getitem_260 * 1.0
        getitem_260 = None
        arange_87 = torch.arange(14)
        getitem_261 = arange_87[(None, slice(None, None, None))]
        arange_87 = None
        k_coords_43 = getitem_261 * 1.0
        getitem_261 = None
        sub_43 = q_coords_43 - k_coords_43
        q_coords_43 = k_coords_43 = None
        relative_coords_43 = sub_43 + 13.0
        sub_43 = None
        long_43 = relative_coords_43.long()
        relative_coords_43 = None
        Rw_21 = l_self_modules_blocks_modules_21_modules_attn_parameters_rel_pos_w_[
            long_43
        ]
        l_self_modules_blocks_modules_21_modules_attn_parameters_rel_pos_w_ = (
            long_43
        ) = None
        r_q_21 = q_21.reshape(16, 14, 14, 80)
        rel_h_21 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_21, Rh_21)
        Rh_21 = None
        rel_w_21 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_21, Rw_21)
        r_q_21 = Rw_21 = None
        getitem_263 = rel_h_21[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_21 = None
        getitem_264 = rel_w_21[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_21 = None
        attn_bias_42 = getitem_263 + getitem_264
        getitem_263 = getitem_264 = None
        attn_bias_43 = attn_bias_42.reshape(-1, 196, 196)
        attn_bias_42 = None
        x_439 = torch._C._nn.scaled_dot_product_attention(
            q_21, k_21, v_21, attn_mask=attn_bias_43, dropout_p=0.0
        )
        q_21 = k_21 = v_21 = attn_bias_43 = None
        view_142 = x_439.view(1, 16, 196, -1)
        x_439 = None
        transpose_21 = view_142.transpose(1, 2)
        view_142 = None
        x_440 = transpose_21.reshape(1, 196, -1)
        transpose_21 = None
        x_441 = torch._C._nn.linear(
            x_440,
            l_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_bias_,
        )
        x_440 = l_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_442 = torch.nn.functional.dropout(x_441, 0.0, False, False)
        x_441 = None
        x_443 = x_442.view(1, 14, 14, -1)
        x_442 = None
        x_444 = x_443.view(1, 1, 1, 14, 14, -1)
        x_443 = None
        permute_72 = x_444.permute(0, 1, 3, 2, 4, 5)
        x_444 = None
        contiguous_58 = permute_72.contiguous()
        permute_72 = None
        x_445 = contiguous_58.view(1, 14, 14, -1)
        contiguous_58 = None
        getitem_265 = x_445[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_445 = None
        x_446 = getitem_265.contiguous()
        getitem_265 = None
        x_447 = x_434 + x_446
        x_434 = x_446 = None
        x_448 = x_447.reshape(1, 196, -1)
        x_447 = None
        layer_norm_43 = torch.nn.functional.layer_norm(
            x_448,
            (1280,),
            l_self_modules_blocks_modules_21_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_21_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_21_modules_norm2_parameters_bias_
        ) = None
        x_449 = torch._C._nn.linear(
            layer_norm_43,
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_43 = (
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_450 = torch._C._nn.gelu(x_449, approximate="none")
        x_449 = None
        x_451 = torch.nn.functional.dropout(x_450, 0.0, False, False)
        x_450 = None
        x_452 = torch._C._nn.linear(
            x_451,
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_451 = (
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_453 = torch.nn.functional.dropout(x_452, 0.0, False, False)
        x_452 = None
        x_454 = x_448 + x_453
        x_448 = x_453 = None
        x_455 = x_454.reshape(1, 14, 14, -1)
        x_454 = None
        x_456 = torch.nn.functional.layer_norm(
            x_455,
            (1280,),
            l_self_modules_blocks_modules_22_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_22_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_22_modules_norm1_parameters_bias_
        ) = None
        x_457 = torch._C._nn.pad(x_456, (0, 0, 0, 0, 0, 0), "constant", None)
        x_456 = None
        x_458 = x_457.view(1, 1, 14, 1, 14, 1280)
        x_457 = None
        permute_73 = x_458.permute(0, 1, 3, 2, 4, 5)
        x_458 = None
        contiguous_60 = permute_73.contiguous()
        permute_73 = None
        windows_20 = contiguous_60.view(-1, 14, 14, 1280)
        contiguous_60 = None
        x_459 = windows_20.reshape(1, 196, -1)
        windows_20 = None
        linear_88 = torch._C._nn.linear(
            x_459,
            l_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_bias_,
        )
        x_459 = (
            l_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_148 = linear_88.view(1, 196, 3, 16, -1)
        linear_88 = None
        qkv_22 = view_148.permute(2, 0, 3, 1, 4)
        view_148 = None
        reshape_164 = qkv_22.reshape(3, 16, 196, -1)
        qkv_22 = None
        unbind_22 = reshape_164.unbind(0)
        reshape_164 = None
        q_22 = unbind_22[0]
        k_22 = unbind_22[1]
        v_22 = unbind_22[2]
        unbind_22 = None
        arange_88 = torch.arange(14)
        getitem_269 = arange_88[(slice(None, None, None), None)]
        arange_88 = None
        q_coords_44 = getitem_269 * 1.0
        getitem_269 = None
        arange_89 = torch.arange(14)
        getitem_270 = arange_89[(None, slice(None, None, None))]
        arange_89 = None
        k_coords_44 = getitem_270 * 1.0
        getitem_270 = None
        sub_44 = q_coords_44 - k_coords_44
        q_coords_44 = k_coords_44 = None
        relative_coords_44 = sub_44 + 13.0
        sub_44 = None
        long_44 = relative_coords_44.long()
        relative_coords_44 = None
        Rh_22 = l_self_modules_blocks_modules_22_modules_attn_parameters_rel_pos_h_[
            long_44
        ]
        l_self_modules_blocks_modules_22_modules_attn_parameters_rel_pos_h_ = (
            long_44
        ) = None
        arange_90 = torch.arange(14)
        getitem_272 = arange_90[(slice(None, None, None), None)]
        arange_90 = None
        q_coords_45 = getitem_272 * 1.0
        getitem_272 = None
        arange_91 = torch.arange(14)
        getitem_273 = arange_91[(None, slice(None, None, None))]
        arange_91 = None
        k_coords_45 = getitem_273 * 1.0
        getitem_273 = None
        sub_45 = q_coords_45 - k_coords_45
        q_coords_45 = k_coords_45 = None
        relative_coords_45 = sub_45 + 13.0
        sub_45 = None
        long_45 = relative_coords_45.long()
        relative_coords_45 = None
        Rw_22 = l_self_modules_blocks_modules_22_modules_attn_parameters_rel_pos_w_[
            long_45
        ]
        l_self_modules_blocks_modules_22_modules_attn_parameters_rel_pos_w_ = (
            long_45
        ) = None
        r_q_22 = q_22.reshape(16, 14, 14, 80)
        rel_h_22 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_22, Rh_22)
        Rh_22 = None
        rel_w_22 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_22, Rw_22)
        r_q_22 = Rw_22 = None
        getitem_275 = rel_h_22[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_22 = None
        getitem_276 = rel_w_22[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_22 = None
        attn_bias_44 = getitem_275 + getitem_276
        getitem_275 = getitem_276 = None
        attn_bias_45 = attn_bias_44.reshape(-1, 196, 196)
        attn_bias_44 = None
        x_460 = torch._C._nn.scaled_dot_product_attention(
            q_22, k_22, v_22, attn_mask=attn_bias_45, dropout_p=0.0
        )
        q_22 = k_22 = v_22 = attn_bias_45 = None
        view_149 = x_460.view(1, 16, 196, -1)
        x_460 = None
        transpose_22 = view_149.transpose(1, 2)
        view_149 = None
        x_461 = transpose_22.reshape(1, 196, -1)
        transpose_22 = None
        x_462 = torch._C._nn.linear(
            x_461,
            l_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_bias_,
        )
        x_461 = l_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_463 = torch.nn.functional.dropout(x_462, 0.0, False, False)
        x_462 = None
        x_464 = x_463.view(1, 14, 14, -1)
        x_463 = None
        x_465 = x_464.view(1, 1, 1, 14, 14, -1)
        x_464 = None
        permute_75 = x_465.permute(0, 1, 3, 2, 4, 5)
        x_465 = None
        contiguous_61 = permute_75.contiguous()
        permute_75 = None
        x_466 = contiguous_61.view(1, 14, 14, -1)
        contiguous_61 = None
        getitem_277 = x_466[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_466 = None
        x_467 = getitem_277.contiguous()
        getitem_277 = None
        x_468 = x_455 + x_467
        x_455 = x_467 = None
        x_469 = x_468.reshape(1, 196, -1)
        x_468 = None
        layer_norm_45 = torch.nn.functional.layer_norm(
            x_469,
            (1280,),
            l_self_modules_blocks_modules_22_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_22_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_22_modules_norm2_parameters_bias_
        ) = None
        x_470 = torch._C._nn.linear(
            layer_norm_45,
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_45 = (
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_471 = torch._C._nn.gelu(x_470, approximate="none")
        x_470 = None
        x_472 = torch.nn.functional.dropout(x_471, 0.0, False, False)
        x_471 = None
        x_473 = torch._C._nn.linear(
            x_472,
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_472 = (
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_474 = torch.nn.functional.dropout(x_473, 0.0, False, False)
        x_473 = None
        x_475 = x_469 + x_474
        x_469 = x_474 = None
        x_476 = x_475.reshape(1, 14, 14, -1)
        x_475 = None
        x_477 = torch.nn.functional.layer_norm(
            x_476,
            (1280,),
            l_self_modules_blocks_modules_23_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_23_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_23_modules_norm1_parameters_bias_
        ) = None
        x_478 = x_477.reshape(1, 196, -1)
        x_477 = None
        linear_92 = torch._C._nn.linear(
            x_478,
            l_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_bias_,
        )
        x_478 = (
            l_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_153 = linear_92.view(1, 196, 3, 16, -1)
        linear_92 = None
        qkv_23 = view_153.permute(2, 0, 3, 1, 4)
        view_153 = None
        reshape_171 = qkv_23.reshape(3, 16, 196, -1)
        qkv_23 = None
        unbind_23 = reshape_171.unbind(0)
        reshape_171 = None
        q_23 = unbind_23[0]
        k_23 = unbind_23[1]
        v_23 = unbind_23[2]
        unbind_23 = None
        reshape_172 = (
            l_self_modules_blocks_modules_23_modules_attn_parameters_rel_pos_h_.reshape(
                1, 127, -1
            )
        )
        l_self_modules_blocks_modules_23_modules_attn_parameters_rel_pos_h_ = None
        permute_77 = reshape_172.permute(0, 2, 1)
        reshape_172 = None
        rel_pos_resized_8 = torch.nn.functional.interpolate(
            permute_77, size=27, mode="linear"
        )
        permute_77 = None
        reshape_173 = rel_pos_resized_8.reshape(-1, 27)
        rel_pos_resized_8 = None
        rel_pos_resized_9 = reshape_173.permute(1, 0)
        reshape_173 = None
        arange_92 = torch.arange(14)
        getitem_281 = arange_92[(slice(None, None, None), None)]
        arange_92 = None
        q_coords_46 = getitem_281 * 1.0
        getitem_281 = None
        arange_93 = torch.arange(14)
        getitem_282 = arange_93[(None, slice(None, None, None))]
        arange_93 = None
        k_coords_46 = getitem_282 * 1.0
        getitem_282 = None
        sub_46 = q_coords_46 - k_coords_46
        q_coords_46 = k_coords_46 = None
        relative_coords_46 = sub_46 + 13.0
        sub_46 = None
        long_46 = relative_coords_46.long()
        relative_coords_46 = None
        Rh_23 = rel_pos_resized_9[long_46]
        rel_pos_resized_9 = long_46 = None
        reshape_174 = (
            l_self_modules_blocks_modules_23_modules_attn_parameters_rel_pos_w_.reshape(
                1, 127, -1
            )
        )
        l_self_modules_blocks_modules_23_modules_attn_parameters_rel_pos_w_ = None
        permute_79 = reshape_174.permute(0, 2, 1)
        reshape_174 = None
        rel_pos_resized_10 = torch.nn.functional.interpolate(
            permute_79, size=27, mode="linear"
        )
        permute_79 = None
        reshape_175 = rel_pos_resized_10.reshape(-1, 27)
        rel_pos_resized_10 = None
        rel_pos_resized_11 = reshape_175.permute(1, 0)
        reshape_175 = None
        arange_94 = torch.arange(14)
        getitem_284 = arange_94[(slice(None, None, None), None)]
        arange_94 = None
        q_coords_47 = getitem_284 * 1.0
        getitem_284 = None
        arange_95 = torch.arange(14)
        getitem_285 = arange_95[(None, slice(None, None, None))]
        arange_95 = None
        k_coords_47 = getitem_285 * 1.0
        getitem_285 = None
        sub_47 = q_coords_47 - k_coords_47
        q_coords_47 = k_coords_47 = None
        relative_coords_47 = sub_47 + 13.0
        sub_47 = None
        long_47 = relative_coords_47.long()
        relative_coords_47 = None
        Rw_23 = rel_pos_resized_11[long_47]
        rel_pos_resized_11 = long_47 = None
        r_q_23 = q_23.reshape(16, 14, 14, 80)
        rel_h_23 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_23, Rh_23)
        Rh_23 = None
        rel_w_23 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_23, Rw_23)
        r_q_23 = Rw_23 = None
        getitem_287 = rel_h_23[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_23 = None
        getitem_288 = rel_w_23[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_23 = None
        attn_bias_46 = getitem_287 + getitem_288
        getitem_287 = getitem_288 = None
        attn_bias_47 = attn_bias_46.reshape(-1, 196, 196)
        attn_bias_46 = None
        x_479 = torch._C._nn.scaled_dot_product_attention(
            q_23, k_23, v_23, attn_mask=attn_bias_47, dropout_p=0.0
        )
        q_23 = k_23 = v_23 = attn_bias_47 = None
        view_154 = x_479.view(1, 16, 196, -1)
        x_479 = None
        transpose_23 = view_154.transpose(1, 2)
        view_154 = None
        x_480 = transpose_23.reshape(1, 196, -1)
        transpose_23 = None
        x_481 = torch._C._nn.linear(
            x_480,
            l_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_bias_,
        )
        x_480 = l_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_482 = torch.nn.functional.dropout(x_481, 0.0, False, False)
        x_481 = None
        x_483 = x_482.view(1, 14, 14, -1)
        x_482 = None
        x_484 = x_476 + x_483
        x_476 = x_483 = None
        x_485 = x_484.reshape(1, 196, -1)
        x_484 = None
        layer_norm_47 = torch.nn.functional.layer_norm(
            x_485,
            (1280,),
            l_self_modules_blocks_modules_23_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_23_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_23_modules_norm2_parameters_bias_
        ) = None
        x_486 = torch._C._nn.linear(
            layer_norm_47,
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_47 = (
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_487 = torch._C._nn.gelu(x_486, approximate="none")
        x_486 = None
        x_488 = torch.nn.functional.dropout(x_487, 0.0, False, False)
        x_487 = None
        x_489 = torch._C._nn.linear(
            x_488,
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_488 = (
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_490 = torch.nn.functional.dropout(x_489, 0.0, False, False)
        x_489 = None
        x_491 = x_485 + x_490
        x_485 = x_490 = None
        x_492 = x_491.reshape(1, 14, 14, -1)
        x_491 = None
        x_493 = torch.nn.functional.layer_norm(
            x_492,
            (1280,),
            l_self_modules_blocks_modules_24_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_24_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_24_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_24_modules_norm1_parameters_bias_
        ) = None
        x_494 = torch._C._nn.pad(x_493, (0, 0, 0, 0, 0, 0), "constant", None)
        x_493 = None
        x_495 = x_494.view(1, 1, 14, 1, 14, 1280)
        x_494 = None
        permute_81 = x_495.permute(0, 1, 3, 2, 4, 5)
        x_495 = None
        contiguous_63 = permute_81.contiguous()
        permute_81 = None
        windows_21 = contiguous_63.view(-1, 14, 14, 1280)
        contiguous_63 = None
        x_496 = windows_21.reshape(1, 196, -1)
        windows_21 = None
        linear_96 = torch._C._nn.linear(
            x_496,
            l_self_modules_blocks_modules_24_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_24_modules_attn_modules_qkv_parameters_bias_,
        )
        x_496 = (
            l_self_modules_blocks_modules_24_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_24_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_158 = linear_96.view(1, 196, 3, 16, -1)
        linear_96 = None
        qkv_24 = view_158.permute(2, 0, 3, 1, 4)
        view_158 = None
        reshape_182 = qkv_24.reshape(3, 16, 196, -1)
        qkv_24 = None
        unbind_24 = reshape_182.unbind(0)
        reshape_182 = None
        q_24 = unbind_24[0]
        k_24 = unbind_24[1]
        v_24 = unbind_24[2]
        unbind_24 = None
        arange_96 = torch.arange(14)
        getitem_292 = arange_96[(slice(None, None, None), None)]
        arange_96 = None
        q_coords_48 = getitem_292 * 1.0
        getitem_292 = None
        arange_97 = torch.arange(14)
        getitem_293 = arange_97[(None, slice(None, None, None))]
        arange_97 = None
        k_coords_48 = getitem_293 * 1.0
        getitem_293 = None
        sub_48 = q_coords_48 - k_coords_48
        q_coords_48 = k_coords_48 = None
        relative_coords_48 = sub_48 + 13.0
        sub_48 = None
        long_48 = relative_coords_48.long()
        relative_coords_48 = None
        Rh_24 = l_self_modules_blocks_modules_24_modules_attn_parameters_rel_pos_h_[
            long_48
        ]
        l_self_modules_blocks_modules_24_modules_attn_parameters_rel_pos_h_ = (
            long_48
        ) = None
        arange_98 = torch.arange(14)
        getitem_295 = arange_98[(slice(None, None, None), None)]
        arange_98 = None
        q_coords_49 = getitem_295 * 1.0
        getitem_295 = None
        arange_99 = torch.arange(14)
        getitem_296 = arange_99[(None, slice(None, None, None))]
        arange_99 = None
        k_coords_49 = getitem_296 * 1.0
        getitem_296 = None
        sub_49 = q_coords_49 - k_coords_49
        q_coords_49 = k_coords_49 = None
        relative_coords_49 = sub_49 + 13.0
        sub_49 = None
        long_49 = relative_coords_49.long()
        relative_coords_49 = None
        Rw_24 = l_self_modules_blocks_modules_24_modules_attn_parameters_rel_pos_w_[
            long_49
        ]
        l_self_modules_blocks_modules_24_modules_attn_parameters_rel_pos_w_ = (
            long_49
        ) = None
        r_q_24 = q_24.reshape(16, 14, 14, 80)
        rel_h_24 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_24, Rh_24)
        Rh_24 = None
        rel_w_24 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_24, Rw_24)
        r_q_24 = Rw_24 = None
        getitem_298 = rel_h_24[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_24 = None
        getitem_299 = rel_w_24[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_24 = None
        attn_bias_48 = getitem_298 + getitem_299
        getitem_298 = getitem_299 = None
        attn_bias_49 = attn_bias_48.reshape(-1, 196, 196)
        attn_bias_48 = None
        x_497 = torch._C._nn.scaled_dot_product_attention(
            q_24, k_24, v_24, attn_mask=attn_bias_49, dropout_p=0.0
        )
        q_24 = k_24 = v_24 = attn_bias_49 = None
        view_159 = x_497.view(1, 16, 196, -1)
        x_497 = None
        transpose_24 = view_159.transpose(1, 2)
        view_159 = None
        x_498 = transpose_24.reshape(1, 196, -1)
        transpose_24 = None
        x_499 = torch._C._nn.linear(
            x_498,
            l_self_modules_blocks_modules_24_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_24_modules_attn_modules_proj_parameters_bias_,
        )
        x_498 = l_self_modules_blocks_modules_24_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_24_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_500 = torch.nn.functional.dropout(x_499, 0.0, False, False)
        x_499 = None
        x_501 = x_500.view(1, 14, 14, -1)
        x_500 = None
        x_502 = x_501.view(1, 1, 1, 14, 14, -1)
        x_501 = None
        permute_83 = x_502.permute(0, 1, 3, 2, 4, 5)
        x_502 = None
        contiguous_64 = permute_83.contiguous()
        permute_83 = None
        x_503 = contiguous_64.view(1, 14, 14, -1)
        contiguous_64 = None
        getitem_300 = x_503[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_503 = None
        x_504 = getitem_300.contiguous()
        getitem_300 = None
        x_505 = x_492 + x_504
        x_492 = x_504 = None
        x_506 = x_505.reshape(1, 196, -1)
        x_505 = None
        layer_norm_49 = torch.nn.functional.layer_norm(
            x_506,
            (1280,),
            l_self_modules_blocks_modules_24_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_24_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_24_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_24_modules_norm2_parameters_bias_
        ) = None
        x_507 = torch._C._nn.linear(
            layer_norm_49,
            l_self_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_49 = (
            l_self_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_508 = torch._C._nn.gelu(x_507, approximate="none")
        x_507 = None
        x_509 = torch.nn.functional.dropout(x_508, 0.0, False, False)
        x_508 = None
        x_510 = torch._C._nn.linear(
            x_509,
            l_self_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_509 = (
            l_self_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_511 = torch.nn.functional.dropout(x_510, 0.0, False, False)
        x_510 = None
        x_512 = x_506 + x_511
        x_506 = x_511 = None
        x_513 = x_512.reshape(1, 14, 14, -1)
        x_512 = None
        x_514 = torch.nn.functional.layer_norm(
            x_513,
            (1280,),
            l_self_modules_blocks_modules_25_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_25_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_25_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_25_modules_norm1_parameters_bias_
        ) = None
        x_515 = torch._C._nn.pad(x_514, (0, 0, 0, 0, 0, 0), "constant", None)
        x_514 = None
        x_516 = x_515.view(1, 1, 14, 1, 14, 1280)
        x_515 = None
        permute_84 = x_516.permute(0, 1, 3, 2, 4, 5)
        x_516 = None
        contiguous_66 = permute_84.contiguous()
        permute_84 = None
        windows_22 = contiguous_66.view(-1, 14, 14, 1280)
        contiguous_66 = None
        x_517 = windows_22.reshape(1, 196, -1)
        windows_22 = None
        linear_100 = torch._C._nn.linear(
            x_517,
            l_self_modules_blocks_modules_25_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_25_modules_attn_modules_qkv_parameters_bias_,
        )
        x_517 = (
            l_self_modules_blocks_modules_25_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_25_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_165 = linear_100.view(1, 196, 3, 16, -1)
        linear_100 = None
        qkv_25 = view_165.permute(2, 0, 3, 1, 4)
        view_165 = None
        reshape_189 = qkv_25.reshape(3, 16, 196, -1)
        qkv_25 = None
        unbind_25 = reshape_189.unbind(0)
        reshape_189 = None
        q_25 = unbind_25[0]
        k_25 = unbind_25[1]
        v_25 = unbind_25[2]
        unbind_25 = None
        arange_100 = torch.arange(14)
        getitem_304 = arange_100[(slice(None, None, None), None)]
        arange_100 = None
        q_coords_50 = getitem_304 * 1.0
        getitem_304 = None
        arange_101 = torch.arange(14)
        getitem_305 = arange_101[(None, slice(None, None, None))]
        arange_101 = None
        k_coords_50 = getitem_305 * 1.0
        getitem_305 = None
        sub_50 = q_coords_50 - k_coords_50
        q_coords_50 = k_coords_50 = None
        relative_coords_50 = sub_50 + 13.0
        sub_50 = None
        long_50 = relative_coords_50.long()
        relative_coords_50 = None
        Rh_25 = l_self_modules_blocks_modules_25_modules_attn_parameters_rel_pos_h_[
            long_50
        ]
        l_self_modules_blocks_modules_25_modules_attn_parameters_rel_pos_h_ = (
            long_50
        ) = None
        arange_102 = torch.arange(14)
        getitem_307 = arange_102[(slice(None, None, None), None)]
        arange_102 = None
        q_coords_51 = getitem_307 * 1.0
        getitem_307 = None
        arange_103 = torch.arange(14)
        getitem_308 = arange_103[(None, slice(None, None, None))]
        arange_103 = None
        k_coords_51 = getitem_308 * 1.0
        getitem_308 = None
        sub_51 = q_coords_51 - k_coords_51
        q_coords_51 = k_coords_51 = None
        relative_coords_51 = sub_51 + 13.0
        sub_51 = None
        long_51 = relative_coords_51.long()
        relative_coords_51 = None
        Rw_25 = l_self_modules_blocks_modules_25_modules_attn_parameters_rel_pos_w_[
            long_51
        ]
        l_self_modules_blocks_modules_25_modules_attn_parameters_rel_pos_w_ = (
            long_51
        ) = None
        r_q_25 = q_25.reshape(16, 14, 14, 80)
        rel_h_25 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_25, Rh_25)
        Rh_25 = None
        rel_w_25 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_25, Rw_25)
        r_q_25 = Rw_25 = None
        getitem_310 = rel_h_25[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_25 = None
        getitem_311 = rel_w_25[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_25 = None
        attn_bias_50 = getitem_310 + getitem_311
        getitem_310 = getitem_311 = None
        attn_bias_51 = attn_bias_50.reshape(-1, 196, 196)
        attn_bias_50 = None
        x_518 = torch._C._nn.scaled_dot_product_attention(
            q_25, k_25, v_25, attn_mask=attn_bias_51, dropout_p=0.0
        )
        q_25 = k_25 = v_25 = attn_bias_51 = None
        view_166 = x_518.view(1, 16, 196, -1)
        x_518 = None
        transpose_25 = view_166.transpose(1, 2)
        view_166 = None
        x_519 = transpose_25.reshape(1, 196, -1)
        transpose_25 = None
        x_520 = torch._C._nn.linear(
            x_519,
            l_self_modules_blocks_modules_25_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_25_modules_attn_modules_proj_parameters_bias_,
        )
        x_519 = l_self_modules_blocks_modules_25_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_25_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_521 = torch.nn.functional.dropout(x_520, 0.0, False, False)
        x_520 = None
        x_522 = x_521.view(1, 14, 14, -1)
        x_521 = None
        x_523 = x_522.view(1, 1, 1, 14, 14, -1)
        x_522 = None
        permute_86 = x_523.permute(0, 1, 3, 2, 4, 5)
        x_523 = None
        contiguous_67 = permute_86.contiguous()
        permute_86 = None
        x_524 = contiguous_67.view(1, 14, 14, -1)
        contiguous_67 = None
        getitem_312 = x_524[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_524 = None
        x_525 = getitem_312.contiguous()
        getitem_312 = None
        x_526 = x_513 + x_525
        x_513 = x_525 = None
        x_527 = x_526.reshape(1, 196, -1)
        x_526 = None
        layer_norm_51 = torch.nn.functional.layer_norm(
            x_527,
            (1280,),
            l_self_modules_blocks_modules_25_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_25_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_25_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_25_modules_norm2_parameters_bias_
        ) = None
        x_528 = torch._C._nn.linear(
            layer_norm_51,
            l_self_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_51 = (
            l_self_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_529 = torch._C._nn.gelu(x_528, approximate="none")
        x_528 = None
        x_530 = torch.nn.functional.dropout(x_529, 0.0, False, False)
        x_529 = None
        x_531 = torch._C._nn.linear(
            x_530,
            l_self_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_530 = (
            l_self_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_532 = torch.nn.functional.dropout(x_531, 0.0, False, False)
        x_531 = None
        x_533 = x_527 + x_532
        x_527 = x_532 = None
        x_534 = x_533.reshape(1, 14, 14, -1)
        x_533 = None
        x_535 = torch.nn.functional.layer_norm(
            x_534,
            (1280,),
            l_self_modules_blocks_modules_26_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_26_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_26_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_26_modules_norm1_parameters_bias_
        ) = None
        x_536 = torch._C._nn.pad(x_535, (0, 0, 0, 0, 0, 0), "constant", None)
        x_535 = None
        x_537 = x_536.view(1, 1, 14, 1, 14, 1280)
        x_536 = None
        permute_87 = x_537.permute(0, 1, 3, 2, 4, 5)
        x_537 = None
        contiguous_69 = permute_87.contiguous()
        permute_87 = None
        windows_23 = contiguous_69.view(-1, 14, 14, 1280)
        contiguous_69 = None
        x_538 = windows_23.reshape(1, 196, -1)
        windows_23 = None
        linear_104 = torch._C._nn.linear(
            x_538,
            l_self_modules_blocks_modules_26_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_26_modules_attn_modules_qkv_parameters_bias_,
        )
        x_538 = (
            l_self_modules_blocks_modules_26_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_26_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_172 = linear_104.view(1, 196, 3, 16, -1)
        linear_104 = None
        qkv_26 = view_172.permute(2, 0, 3, 1, 4)
        view_172 = None
        reshape_196 = qkv_26.reshape(3, 16, 196, -1)
        qkv_26 = None
        unbind_26 = reshape_196.unbind(0)
        reshape_196 = None
        q_26 = unbind_26[0]
        k_26 = unbind_26[1]
        v_26 = unbind_26[2]
        unbind_26 = None
        arange_104 = torch.arange(14)
        getitem_316 = arange_104[(slice(None, None, None), None)]
        arange_104 = None
        q_coords_52 = getitem_316 * 1.0
        getitem_316 = None
        arange_105 = torch.arange(14)
        getitem_317 = arange_105[(None, slice(None, None, None))]
        arange_105 = None
        k_coords_52 = getitem_317 * 1.0
        getitem_317 = None
        sub_52 = q_coords_52 - k_coords_52
        q_coords_52 = k_coords_52 = None
        relative_coords_52 = sub_52 + 13.0
        sub_52 = None
        long_52 = relative_coords_52.long()
        relative_coords_52 = None
        Rh_26 = l_self_modules_blocks_modules_26_modules_attn_parameters_rel_pos_h_[
            long_52
        ]
        l_self_modules_blocks_modules_26_modules_attn_parameters_rel_pos_h_ = (
            long_52
        ) = None
        arange_106 = torch.arange(14)
        getitem_319 = arange_106[(slice(None, None, None), None)]
        arange_106 = None
        q_coords_53 = getitem_319 * 1.0
        getitem_319 = None
        arange_107 = torch.arange(14)
        getitem_320 = arange_107[(None, slice(None, None, None))]
        arange_107 = None
        k_coords_53 = getitem_320 * 1.0
        getitem_320 = None
        sub_53 = q_coords_53 - k_coords_53
        q_coords_53 = k_coords_53 = None
        relative_coords_53 = sub_53 + 13.0
        sub_53 = None
        long_53 = relative_coords_53.long()
        relative_coords_53 = None
        Rw_26 = l_self_modules_blocks_modules_26_modules_attn_parameters_rel_pos_w_[
            long_53
        ]
        l_self_modules_blocks_modules_26_modules_attn_parameters_rel_pos_w_ = (
            long_53
        ) = None
        r_q_26 = q_26.reshape(16, 14, 14, 80)
        rel_h_26 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_26, Rh_26)
        Rh_26 = None
        rel_w_26 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_26, Rw_26)
        r_q_26 = Rw_26 = None
        getitem_322 = rel_h_26[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_26 = None
        getitem_323 = rel_w_26[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_26 = None
        attn_bias_52 = getitem_322 + getitem_323
        getitem_322 = getitem_323 = None
        attn_bias_53 = attn_bias_52.reshape(-1, 196, 196)
        attn_bias_52 = None
        x_539 = torch._C._nn.scaled_dot_product_attention(
            q_26, k_26, v_26, attn_mask=attn_bias_53, dropout_p=0.0
        )
        q_26 = k_26 = v_26 = attn_bias_53 = None
        view_173 = x_539.view(1, 16, 196, -1)
        x_539 = None
        transpose_26 = view_173.transpose(1, 2)
        view_173 = None
        x_540 = transpose_26.reshape(1, 196, -1)
        transpose_26 = None
        x_541 = torch._C._nn.linear(
            x_540,
            l_self_modules_blocks_modules_26_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_26_modules_attn_modules_proj_parameters_bias_,
        )
        x_540 = l_self_modules_blocks_modules_26_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_26_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_542 = torch.nn.functional.dropout(x_541, 0.0, False, False)
        x_541 = None
        x_543 = x_542.view(1, 14, 14, -1)
        x_542 = None
        x_544 = x_543.view(1, 1, 1, 14, 14, -1)
        x_543 = None
        permute_89 = x_544.permute(0, 1, 3, 2, 4, 5)
        x_544 = None
        contiguous_70 = permute_89.contiguous()
        permute_89 = None
        x_545 = contiguous_70.view(1, 14, 14, -1)
        contiguous_70 = None
        getitem_324 = x_545[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_545 = None
        x_546 = getitem_324.contiguous()
        getitem_324 = None
        x_547 = x_534 + x_546
        x_534 = x_546 = None
        x_548 = x_547.reshape(1, 196, -1)
        x_547 = None
        layer_norm_53 = torch.nn.functional.layer_norm(
            x_548,
            (1280,),
            l_self_modules_blocks_modules_26_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_26_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_26_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_26_modules_norm2_parameters_bias_
        ) = None
        x_549 = torch._C._nn.linear(
            layer_norm_53,
            l_self_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_53 = (
            l_self_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_550 = torch._C._nn.gelu(x_549, approximate="none")
        x_549 = None
        x_551 = torch.nn.functional.dropout(x_550, 0.0, False, False)
        x_550 = None
        x_552 = torch._C._nn.linear(
            x_551,
            l_self_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_551 = (
            l_self_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_553 = torch.nn.functional.dropout(x_552, 0.0, False, False)
        x_552 = None
        x_554 = x_548 + x_553
        x_548 = x_553 = None
        x_555 = x_554.reshape(1, 14, 14, -1)
        x_554 = None
        x_556 = torch.nn.functional.layer_norm(
            x_555,
            (1280,),
            l_self_modules_blocks_modules_27_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_27_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_27_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_27_modules_norm1_parameters_bias_
        ) = None
        x_557 = torch._C._nn.pad(x_556, (0, 0, 0, 0, 0, 0), "constant", None)
        x_556 = None
        x_558 = x_557.view(1, 1, 14, 1, 14, 1280)
        x_557 = None
        permute_90 = x_558.permute(0, 1, 3, 2, 4, 5)
        x_558 = None
        contiguous_72 = permute_90.contiguous()
        permute_90 = None
        windows_24 = contiguous_72.view(-1, 14, 14, 1280)
        contiguous_72 = None
        x_559 = windows_24.reshape(1, 196, -1)
        windows_24 = None
        linear_108 = torch._C._nn.linear(
            x_559,
            l_self_modules_blocks_modules_27_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_27_modules_attn_modules_qkv_parameters_bias_,
        )
        x_559 = (
            l_self_modules_blocks_modules_27_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_27_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_179 = linear_108.view(1, 196, 3, 16, -1)
        linear_108 = None
        qkv_27 = view_179.permute(2, 0, 3, 1, 4)
        view_179 = None
        reshape_203 = qkv_27.reshape(3, 16, 196, -1)
        qkv_27 = None
        unbind_27 = reshape_203.unbind(0)
        reshape_203 = None
        q_27 = unbind_27[0]
        k_27 = unbind_27[1]
        v_27 = unbind_27[2]
        unbind_27 = None
        arange_108 = torch.arange(14)
        getitem_328 = arange_108[(slice(None, None, None), None)]
        arange_108 = None
        q_coords_54 = getitem_328 * 1.0
        getitem_328 = None
        arange_109 = torch.arange(14)
        getitem_329 = arange_109[(None, slice(None, None, None))]
        arange_109 = None
        k_coords_54 = getitem_329 * 1.0
        getitem_329 = None
        sub_54 = q_coords_54 - k_coords_54
        q_coords_54 = k_coords_54 = None
        relative_coords_54 = sub_54 + 13.0
        sub_54 = None
        long_54 = relative_coords_54.long()
        relative_coords_54 = None
        Rh_27 = l_self_modules_blocks_modules_27_modules_attn_parameters_rel_pos_h_[
            long_54
        ]
        l_self_modules_blocks_modules_27_modules_attn_parameters_rel_pos_h_ = (
            long_54
        ) = None
        arange_110 = torch.arange(14)
        getitem_331 = arange_110[(slice(None, None, None), None)]
        arange_110 = None
        q_coords_55 = getitem_331 * 1.0
        getitem_331 = None
        arange_111 = torch.arange(14)
        getitem_332 = arange_111[(None, slice(None, None, None))]
        arange_111 = None
        k_coords_55 = getitem_332 * 1.0
        getitem_332 = None
        sub_55 = q_coords_55 - k_coords_55
        q_coords_55 = k_coords_55 = None
        relative_coords_55 = sub_55 + 13.0
        sub_55 = None
        long_55 = relative_coords_55.long()
        relative_coords_55 = None
        Rw_27 = l_self_modules_blocks_modules_27_modules_attn_parameters_rel_pos_w_[
            long_55
        ]
        l_self_modules_blocks_modules_27_modules_attn_parameters_rel_pos_w_ = (
            long_55
        ) = None
        r_q_27 = q_27.reshape(16, 14, 14, 80)
        rel_h_27 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_27, Rh_27)
        Rh_27 = None
        rel_w_27 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_27, Rw_27)
        r_q_27 = Rw_27 = None
        getitem_334 = rel_h_27[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_27 = None
        getitem_335 = rel_w_27[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_27 = None
        attn_bias_54 = getitem_334 + getitem_335
        getitem_334 = getitem_335 = None
        attn_bias_55 = attn_bias_54.reshape(-1, 196, 196)
        attn_bias_54 = None
        x_560 = torch._C._nn.scaled_dot_product_attention(
            q_27, k_27, v_27, attn_mask=attn_bias_55, dropout_p=0.0
        )
        q_27 = k_27 = v_27 = attn_bias_55 = None
        view_180 = x_560.view(1, 16, 196, -1)
        x_560 = None
        transpose_27 = view_180.transpose(1, 2)
        view_180 = None
        x_561 = transpose_27.reshape(1, 196, -1)
        transpose_27 = None
        x_562 = torch._C._nn.linear(
            x_561,
            l_self_modules_blocks_modules_27_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_27_modules_attn_modules_proj_parameters_bias_,
        )
        x_561 = l_self_modules_blocks_modules_27_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_27_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_563 = torch.nn.functional.dropout(x_562, 0.0, False, False)
        x_562 = None
        x_564 = x_563.view(1, 14, 14, -1)
        x_563 = None
        x_565 = x_564.view(1, 1, 1, 14, 14, -1)
        x_564 = None
        permute_92 = x_565.permute(0, 1, 3, 2, 4, 5)
        x_565 = None
        contiguous_73 = permute_92.contiguous()
        permute_92 = None
        x_566 = contiguous_73.view(1, 14, 14, -1)
        contiguous_73 = None
        getitem_336 = x_566[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_566 = None
        x_567 = getitem_336.contiguous()
        getitem_336 = None
        x_568 = x_555 + x_567
        x_555 = x_567 = None
        x_569 = x_568.reshape(1, 196, -1)
        x_568 = None
        layer_norm_55 = torch.nn.functional.layer_norm(
            x_569,
            (1280,),
            l_self_modules_blocks_modules_27_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_27_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_27_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_27_modules_norm2_parameters_bias_
        ) = None
        x_570 = torch._C._nn.linear(
            layer_norm_55,
            l_self_modules_blocks_modules_27_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_27_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_55 = (
            l_self_modules_blocks_modules_27_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_27_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_571 = torch._C._nn.gelu(x_570, approximate="none")
        x_570 = None
        x_572 = torch.nn.functional.dropout(x_571, 0.0, False, False)
        x_571 = None
        x_573 = torch._C._nn.linear(
            x_572,
            l_self_modules_blocks_modules_27_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_27_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_572 = (
            l_self_modules_blocks_modules_27_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_27_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_574 = torch.nn.functional.dropout(x_573, 0.0, False, False)
        x_573 = None
        x_575 = x_569 + x_574
        x_569 = x_574 = None
        x_576 = x_575.reshape(1, 14, 14, -1)
        x_575 = None
        x_577 = torch.nn.functional.layer_norm(
            x_576,
            (1280,),
            l_self_modules_blocks_modules_28_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_28_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_28_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_28_modules_norm1_parameters_bias_
        ) = None
        x_578 = torch._C._nn.pad(x_577, (0, 0, 0, 0, 0, 0), "constant", None)
        x_577 = None
        x_579 = x_578.view(1, 1, 14, 1, 14, 1280)
        x_578 = None
        permute_93 = x_579.permute(0, 1, 3, 2, 4, 5)
        x_579 = None
        contiguous_75 = permute_93.contiguous()
        permute_93 = None
        windows_25 = contiguous_75.view(-1, 14, 14, 1280)
        contiguous_75 = None
        x_580 = windows_25.reshape(1, 196, -1)
        windows_25 = None
        linear_112 = torch._C._nn.linear(
            x_580,
            l_self_modules_blocks_modules_28_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_28_modules_attn_modules_qkv_parameters_bias_,
        )
        x_580 = (
            l_self_modules_blocks_modules_28_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_28_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_186 = linear_112.view(1, 196, 3, 16, -1)
        linear_112 = None
        qkv_28 = view_186.permute(2, 0, 3, 1, 4)
        view_186 = None
        reshape_210 = qkv_28.reshape(3, 16, 196, -1)
        qkv_28 = None
        unbind_28 = reshape_210.unbind(0)
        reshape_210 = None
        q_28 = unbind_28[0]
        k_28 = unbind_28[1]
        v_28 = unbind_28[2]
        unbind_28 = None
        arange_112 = torch.arange(14)
        getitem_340 = arange_112[(slice(None, None, None), None)]
        arange_112 = None
        q_coords_56 = getitem_340 * 1.0
        getitem_340 = None
        arange_113 = torch.arange(14)
        getitem_341 = arange_113[(None, slice(None, None, None))]
        arange_113 = None
        k_coords_56 = getitem_341 * 1.0
        getitem_341 = None
        sub_56 = q_coords_56 - k_coords_56
        q_coords_56 = k_coords_56 = None
        relative_coords_56 = sub_56 + 13.0
        sub_56 = None
        long_56 = relative_coords_56.long()
        relative_coords_56 = None
        Rh_28 = l_self_modules_blocks_modules_28_modules_attn_parameters_rel_pos_h_[
            long_56
        ]
        l_self_modules_blocks_modules_28_modules_attn_parameters_rel_pos_h_ = (
            long_56
        ) = None
        arange_114 = torch.arange(14)
        getitem_343 = arange_114[(slice(None, None, None), None)]
        arange_114 = None
        q_coords_57 = getitem_343 * 1.0
        getitem_343 = None
        arange_115 = torch.arange(14)
        getitem_344 = arange_115[(None, slice(None, None, None))]
        arange_115 = None
        k_coords_57 = getitem_344 * 1.0
        getitem_344 = None
        sub_57 = q_coords_57 - k_coords_57
        q_coords_57 = k_coords_57 = None
        relative_coords_57 = sub_57 + 13.0
        sub_57 = None
        long_57 = relative_coords_57.long()
        relative_coords_57 = None
        Rw_28 = l_self_modules_blocks_modules_28_modules_attn_parameters_rel_pos_w_[
            long_57
        ]
        l_self_modules_blocks_modules_28_modules_attn_parameters_rel_pos_w_ = (
            long_57
        ) = None
        r_q_28 = q_28.reshape(16, 14, 14, 80)
        rel_h_28 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_28, Rh_28)
        Rh_28 = None
        rel_w_28 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_28, Rw_28)
        r_q_28 = Rw_28 = None
        getitem_346 = rel_h_28[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_28 = None
        getitem_347 = rel_w_28[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_28 = None
        attn_bias_56 = getitem_346 + getitem_347
        getitem_346 = getitem_347 = None
        attn_bias_57 = attn_bias_56.reshape(-1, 196, 196)
        attn_bias_56 = None
        x_581 = torch._C._nn.scaled_dot_product_attention(
            q_28, k_28, v_28, attn_mask=attn_bias_57, dropout_p=0.0
        )
        q_28 = k_28 = v_28 = attn_bias_57 = None
        view_187 = x_581.view(1, 16, 196, -1)
        x_581 = None
        transpose_28 = view_187.transpose(1, 2)
        view_187 = None
        x_582 = transpose_28.reshape(1, 196, -1)
        transpose_28 = None
        x_583 = torch._C._nn.linear(
            x_582,
            l_self_modules_blocks_modules_28_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_28_modules_attn_modules_proj_parameters_bias_,
        )
        x_582 = l_self_modules_blocks_modules_28_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_28_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_584 = torch.nn.functional.dropout(x_583, 0.0, False, False)
        x_583 = None
        x_585 = x_584.view(1, 14, 14, -1)
        x_584 = None
        x_586 = x_585.view(1, 1, 1, 14, 14, -1)
        x_585 = None
        permute_95 = x_586.permute(0, 1, 3, 2, 4, 5)
        x_586 = None
        contiguous_76 = permute_95.contiguous()
        permute_95 = None
        x_587 = contiguous_76.view(1, 14, 14, -1)
        contiguous_76 = None
        getitem_348 = x_587[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_587 = None
        x_588 = getitem_348.contiguous()
        getitem_348 = None
        x_589 = x_576 + x_588
        x_576 = x_588 = None
        x_590 = x_589.reshape(1, 196, -1)
        x_589 = None
        layer_norm_57 = torch.nn.functional.layer_norm(
            x_590,
            (1280,),
            l_self_modules_blocks_modules_28_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_28_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_28_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_28_modules_norm2_parameters_bias_
        ) = None
        x_591 = torch._C._nn.linear(
            layer_norm_57,
            l_self_modules_blocks_modules_28_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_28_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_57 = (
            l_self_modules_blocks_modules_28_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_28_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_592 = torch._C._nn.gelu(x_591, approximate="none")
        x_591 = None
        x_593 = torch.nn.functional.dropout(x_592, 0.0, False, False)
        x_592 = None
        x_594 = torch._C._nn.linear(
            x_593,
            l_self_modules_blocks_modules_28_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_28_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_593 = (
            l_self_modules_blocks_modules_28_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_28_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_595 = torch.nn.functional.dropout(x_594, 0.0, False, False)
        x_594 = None
        x_596 = x_590 + x_595
        x_590 = x_595 = None
        x_597 = x_596.reshape(1, 14, 14, -1)
        x_596 = None
        x_598 = torch.nn.functional.layer_norm(
            x_597,
            (1280,),
            l_self_modules_blocks_modules_29_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_29_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_29_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_29_modules_norm1_parameters_bias_
        ) = None
        x_599 = torch._C._nn.pad(x_598, (0, 0, 0, 0, 0, 0), "constant", None)
        x_598 = None
        x_600 = x_599.view(1, 1, 14, 1, 14, 1280)
        x_599 = None
        permute_96 = x_600.permute(0, 1, 3, 2, 4, 5)
        x_600 = None
        contiguous_78 = permute_96.contiguous()
        permute_96 = None
        windows_26 = contiguous_78.view(-1, 14, 14, 1280)
        contiguous_78 = None
        x_601 = windows_26.reshape(1, 196, -1)
        windows_26 = None
        linear_116 = torch._C._nn.linear(
            x_601,
            l_self_modules_blocks_modules_29_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_29_modules_attn_modules_qkv_parameters_bias_,
        )
        x_601 = (
            l_self_modules_blocks_modules_29_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_29_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_193 = linear_116.view(1, 196, 3, 16, -1)
        linear_116 = None
        qkv_29 = view_193.permute(2, 0, 3, 1, 4)
        view_193 = None
        reshape_217 = qkv_29.reshape(3, 16, 196, -1)
        qkv_29 = None
        unbind_29 = reshape_217.unbind(0)
        reshape_217 = None
        q_29 = unbind_29[0]
        k_29 = unbind_29[1]
        v_29 = unbind_29[2]
        unbind_29 = None
        arange_116 = torch.arange(14)
        getitem_352 = arange_116[(slice(None, None, None), None)]
        arange_116 = None
        q_coords_58 = getitem_352 * 1.0
        getitem_352 = None
        arange_117 = torch.arange(14)
        getitem_353 = arange_117[(None, slice(None, None, None))]
        arange_117 = None
        k_coords_58 = getitem_353 * 1.0
        getitem_353 = None
        sub_58 = q_coords_58 - k_coords_58
        q_coords_58 = k_coords_58 = None
        relative_coords_58 = sub_58 + 13.0
        sub_58 = None
        long_58 = relative_coords_58.long()
        relative_coords_58 = None
        Rh_29 = l_self_modules_blocks_modules_29_modules_attn_parameters_rel_pos_h_[
            long_58
        ]
        l_self_modules_blocks_modules_29_modules_attn_parameters_rel_pos_h_ = (
            long_58
        ) = None
        arange_118 = torch.arange(14)
        getitem_355 = arange_118[(slice(None, None, None), None)]
        arange_118 = None
        q_coords_59 = getitem_355 * 1.0
        getitem_355 = None
        arange_119 = torch.arange(14)
        getitem_356 = arange_119[(None, slice(None, None, None))]
        arange_119 = None
        k_coords_59 = getitem_356 * 1.0
        getitem_356 = None
        sub_59 = q_coords_59 - k_coords_59
        q_coords_59 = k_coords_59 = None
        relative_coords_59 = sub_59 + 13.0
        sub_59 = None
        long_59 = relative_coords_59.long()
        relative_coords_59 = None
        Rw_29 = l_self_modules_blocks_modules_29_modules_attn_parameters_rel_pos_w_[
            long_59
        ]
        l_self_modules_blocks_modules_29_modules_attn_parameters_rel_pos_w_ = (
            long_59
        ) = None
        r_q_29 = q_29.reshape(16, 14, 14, 80)
        rel_h_29 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_29, Rh_29)
        Rh_29 = None
        rel_w_29 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_29, Rw_29)
        r_q_29 = Rw_29 = None
        getitem_358 = rel_h_29[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_29 = None
        getitem_359 = rel_w_29[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_29 = None
        attn_bias_58 = getitem_358 + getitem_359
        getitem_358 = getitem_359 = None
        attn_bias_59 = attn_bias_58.reshape(-1, 196, 196)
        attn_bias_58 = None
        x_602 = torch._C._nn.scaled_dot_product_attention(
            q_29, k_29, v_29, attn_mask=attn_bias_59, dropout_p=0.0
        )
        q_29 = k_29 = v_29 = attn_bias_59 = None
        view_194 = x_602.view(1, 16, 196, -1)
        x_602 = None
        transpose_29 = view_194.transpose(1, 2)
        view_194 = None
        x_603 = transpose_29.reshape(1, 196, -1)
        transpose_29 = None
        x_604 = torch._C._nn.linear(
            x_603,
            l_self_modules_blocks_modules_29_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_29_modules_attn_modules_proj_parameters_bias_,
        )
        x_603 = l_self_modules_blocks_modules_29_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_29_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_605 = torch.nn.functional.dropout(x_604, 0.0, False, False)
        x_604 = None
        x_606 = x_605.view(1, 14, 14, -1)
        x_605 = None
        x_607 = x_606.view(1, 1, 1, 14, 14, -1)
        x_606 = None
        permute_98 = x_607.permute(0, 1, 3, 2, 4, 5)
        x_607 = None
        contiguous_79 = permute_98.contiguous()
        permute_98 = None
        x_608 = contiguous_79.view(1, 14, 14, -1)
        contiguous_79 = None
        getitem_360 = x_608[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_608 = None
        x_609 = getitem_360.contiguous()
        getitem_360 = None
        x_610 = x_597 + x_609
        x_597 = x_609 = None
        x_611 = x_610.reshape(1, 196, -1)
        x_610 = None
        layer_norm_59 = torch.nn.functional.layer_norm(
            x_611,
            (1280,),
            l_self_modules_blocks_modules_29_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_29_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_29_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_29_modules_norm2_parameters_bias_
        ) = None
        x_612 = torch._C._nn.linear(
            layer_norm_59,
            l_self_modules_blocks_modules_29_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_29_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_59 = (
            l_self_modules_blocks_modules_29_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_29_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_613 = torch._C._nn.gelu(x_612, approximate="none")
        x_612 = None
        x_614 = torch.nn.functional.dropout(x_613, 0.0, False, False)
        x_613 = None
        x_615 = torch._C._nn.linear(
            x_614,
            l_self_modules_blocks_modules_29_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_29_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_614 = (
            l_self_modules_blocks_modules_29_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_29_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_616 = torch.nn.functional.dropout(x_615, 0.0, False, False)
        x_615 = None
        x_617 = x_611 + x_616
        x_611 = x_616 = None
        x_618 = x_617.reshape(1, 14, 14, -1)
        x_617 = None
        x_619 = torch.nn.functional.layer_norm(
            x_618,
            (1280,),
            l_self_modules_blocks_modules_30_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_30_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_30_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_30_modules_norm1_parameters_bias_
        ) = None
        x_620 = torch._C._nn.pad(x_619, (0, 0, 0, 0, 0, 0), "constant", None)
        x_619 = None
        x_621 = x_620.view(1, 1, 14, 1, 14, 1280)
        x_620 = None
        permute_99 = x_621.permute(0, 1, 3, 2, 4, 5)
        x_621 = None
        contiguous_81 = permute_99.contiguous()
        permute_99 = None
        windows_27 = contiguous_81.view(-1, 14, 14, 1280)
        contiguous_81 = None
        x_622 = windows_27.reshape(1, 196, -1)
        windows_27 = None
        linear_120 = torch._C._nn.linear(
            x_622,
            l_self_modules_blocks_modules_30_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_30_modules_attn_modules_qkv_parameters_bias_,
        )
        x_622 = (
            l_self_modules_blocks_modules_30_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_30_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_200 = linear_120.view(1, 196, 3, 16, -1)
        linear_120 = None
        qkv_30 = view_200.permute(2, 0, 3, 1, 4)
        view_200 = None
        reshape_224 = qkv_30.reshape(3, 16, 196, -1)
        qkv_30 = None
        unbind_30 = reshape_224.unbind(0)
        reshape_224 = None
        q_30 = unbind_30[0]
        k_30 = unbind_30[1]
        v_30 = unbind_30[2]
        unbind_30 = None
        arange_120 = torch.arange(14)
        getitem_364 = arange_120[(slice(None, None, None), None)]
        arange_120 = None
        q_coords_60 = getitem_364 * 1.0
        getitem_364 = None
        arange_121 = torch.arange(14)
        getitem_365 = arange_121[(None, slice(None, None, None))]
        arange_121 = None
        k_coords_60 = getitem_365 * 1.0
        getitem_365 = None
        sub_60 = q_coords_60 - k_coords_60
        q_coords_60 = k_coords_60 = None
        relative_coords_60 = sub_60 + 13.0
        sub_60 = None
        long_60 = relative_coords_60.long()
        relative_coords_60 = None
        Rh_30 = l_self_modules_blocks_modules_30_modules_attn_parameters_rel_pos_h_[
            long_60
        ]
        l_self_modules_blocks_modules_30_modules_attn_parameters_rel_pos_h_ = (
            long_60
        ) = None
        arange_122 = torch.arange(14)
        getitem_367 = arange_122[(slice(None, None, None), None)]
        arange_122 = None
        q_coords_61 = getitem_367 * 1.0
        getitem_367 = None
        arange_123 = torch.arange(14)
        getitem_368 = arange_123[(None, slice(None, None, None))]
        arange_123 = None
        k_coords_61 = getitem_368 * 1.0
        getitem_368 = None
        sub_61 = q_coords_61 - k_coords_61
        q_coords_61 = k_coords_61 = None
        relative_coords_61 = sub_61 + 13.0
        sub_61 = None
        long_61 = relative_coords_61.long()
        relative_coords_61 = None
        Rw_30 = l_self_modules_blocks_modules_30_modules_attn_parameters_rel_pos_w_[
            long_61
        ]
        l_self_modules_blocks_modules_30_modules_attn_parameters_rel_pos_w_ = (
            long_61
        ) = None
        r_q_30 = q_30.reshape(16, 14, 14, 80)
        rel_h_30 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_30, Rh_30)
        Rh_30 = None
        rel_w_30 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_30, Rw_30)
        r_q_30 = Rw_30 = None
        getitem_370 = rel_h_30[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_30 = None
        getitem_371 = rel_w_30[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_30 = None
        attn_bias_60 = getitem_370 + getitem_371
        getitem_370 = getitem_371 = None
        attn_bias_61 = attn_bias_60.reshape(-1, 196, 196)
        attn_bias_60 = None
        x_623 = torch._C._nn.scaled_dot_product_attention(
            q_30, k_30, v_30, attn_mask=attn_bias_61, dropout_p=0.0
        )
        q_30 = k_30 = v_30 = attn_bias_61 = None
        view_201 = x_623.view(1, 16, 196, -1)
        x_623 = None
        transpose_30 = view_201.transpose(1, 2)
        view_201 = None
        x_624 = transpose_30.reshape(1, 196, -1)
        transpose_30 = None
        x_625 = torch._C._nn.linear(
            x_624,
            l_self_modules_blocks_modules_30_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_30_modules_attn_modules_proj_parameters_bias_,
        )
        x_624 = l_self_modules_blocks_modules_30_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_30_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_626 = torch.nn.functional.dropout(x_625, 0.0, False, False)
        x_625 = None
        x_627 = x_626.view(1, 14, 14, -1)
        x_626 = None
        x_628 = x_627.view(1, 1, 1, 14, 14, -1)
        x_627 = None
        permute_101 = x_628.permute(0, 1, 3, 2, 4, 5)
        x_628 = None
        contiguous_82 = permute_101.contiguous()
        permute_101 = None
        x_629 = contiguous_82.view(1, 14, 14, -1)
        contiguous_82 = None
        getitem_372 = x_629[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_629 = None
        x_630 = getitem_372.contiguous()
        getitem_372 = None
        x_631 = x_618 + x_630
        x_618 = x_630 = None
        x_632 = x_631.reshape(1, 196, -1)
        x_631 = None
        layer_norm_61 = torch.nn.functional.layer_norm(
            x_632,
            (1280,),
            l_self_modules_blocks_modules_30_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_30_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_30_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_30_modules_norm2_parameters_bias_
        ) = None
        x_633 = torch._C._nn.linear(
            layer_norm_61,
            l_self_modules_blocks_modules_30_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_30_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_61 = (
            l_self_modules_blocks_modules_30_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_30_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_634 = torch._C._nn.gelu(x_633, approximate="none")
        x_633 = None
        x_635 = torch.nn.functional.dropout(x_634, 0.0, False, False)
        x_634 = None
        x_636 = torch._C._nn.linear(
            x_635,
            l_self_modules_blocks_modules_30_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_30_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_635 = (
            l_self_modules_blocks_modules_30_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_30_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_637 = torch.nn.functional.dropout(x_636, 0.0, False, False)
        x_636 = None
        x_638 = x_632 + x_637
        x_632 = x_637 = None
        x_639 = x_638.reshape(1, 14, 14, -1)
        x_638 = None
        x_640 = torch.nn.functional.layer_norm(
            x_639,
            (1280,),
            l_self_modules_blocks_modules_31_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_31_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_31_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_31_modules_norm1_parameters_bias_
        ) = None
        x_641 = x_640.reshape(1, 196, -1)
        x_640 = None
        linear_124 = torch._C._nn.linear(
            x_641,
            l_self_modules_blocks_modules_31_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_31_modules_attn_modules_qkv_parameters_bias_,
        )
        x_641 = (
            l_self_modules_blocks_modules_31_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_31_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_205 = linear_124.view(1, 196, 3, 16, -1)
        linear_124 = None
        qkv_31 = view_205.permute(2, 0, 3, 1, 4)
        view_205 = None
        reshape_231 = qkv_31.reshape(3, 16, 196, -1)
        qkv_31 = None
        unbind_31 = reshape_231.unbind(0)
        reshape_231 = None
        q_31 = unbind_31[0]
        k_31 = unbind_31[1]
        v_31 = unbind_31[2]
        unbind_31 = None
        reshape_232 = (
            l_self_modules_blocks_modules_31_modules_attn_parameters_rel_pos_h_.reshape(
                1, 127, -1
            )
        )
        l_self_modules_blocks_modules_31_modules_attn_parameters_rel_pos_h_ = None
        permute_103 = reshape_232.permute(0, 2, 1)
        reshape_232 = None
        rel_pos_resized_12 = torch.nn.functional.interpolate(
            permute_103, size=27, mode="linear"
        )
        permute_103 = None
        reshape_233 = rel_pos_resized_12.reshape(-1, 27)
        rel_pos_resized_12 = None
        rel_pos_resized_13 = reshape_233.permute(1, 0)
        reshape_233 = None
        arange_124 = torch.arange(14)
        getitem_376 = arange_124[(slice(None, None, None), None)]
        arange_124 = None
        q_coords_62 = getitem_376 * 1.0
        getitem_376 = None
        arange_125 = torch.arange(14)
        getitem_377 = arange_125[(None, slice(None, None, None))]
        arange_125 = None
        k_coords_62 = getitem_377 * 1.0
        getitem_377 = None
        sub_62 = q_coords_62 - k_coords_62
        q_coords_62 = k_coords_62 = None
        relative_coords_62 = sub_62 + 13.0
        sub_62 = None
        long_62 = relative_coords_62.long()
        relative_coords_62 = None
        Rh_31 = rel_pos_resized_13[long_62]
        rel_pos_resized_13 = long_62 = None
        reshape_234 = (
            l_self_modules_blocks_modules_31_modules_attn_parameters_rel_pos_w_.reshape(
                1, 127, -1
            )
        )
        l_self_modules_blocks_modules_31_modules_attn_parameters_rel_pos_w_ = None
        permute_105 = reshape_234.permute(0, 2, 1)
        reshape_234 = None
        rel_pos_resized_14 = torch.nn.functional.interpolate(
            permute_105, size=27, mode="linear"
        )
        permute_105 = None
        reshape_235 = rel_pos_resized_14.reshape(-1, 27)
        rel_pos_resized_14 = None
        rel_pos_resized_15 = reshape_235.permute(1, 0)
        reshape_235 = None
        arange_126 = torch.arange(14)
        getitem_379 = arange_126[(slice(None, None, None), None)]
        arange_126 = None
        q_coords_63 = getitem_379 * 1.0
        getitem_379 = None
        arange_127 = torch.arange(14)
        getitem_380 = arange_127[(None, slice(None, None, None))]
        arange_127 = None
        k_coords_63 = getitem_380 * 1.0
        getitem_380 = None
        sub_63 = q_coords_63 - k_coords_63
        q_coords_63 = k_coords_63 = None
        relative_coords_63 = sub_63 + 13.0
        sub_63 = None
        long_63 = relative_coords_63.long()
        relative_coords_63 = None
        Rw_31 = rel_pos_resized_15[long_63]
        rel_pos_resized_15 = long_63 = None
        r_q_31 = q_31.reshape(16, 14, 14, 80)
        rel_h_31 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_31, Rh_31)
        Rh_31 = None
        rel_w_31 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_31, Rw_31)
        r_q_31 = Rw_31 = None
        getitem_382 = rel_h_31[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_31 = None
        getitem_383 = rel_w_31[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_31 = None
        attn_bias_62 = getitem_382 + getitem_383
        getitem_382 = getitem_383 = None
        attn_bias_63 = attn_bias_62.reshape(-1, 196, 196)
        attn_bias_62 = None
        x_642 = torch._C._nn.scaled_dot_product_attention(
            q_31, k_31, v_31, attn_mask=attn_bias_63, dropout_p=0.0
        )
        q_31 = k_31 = v_31 = attn_bias_63 = None
        view_206 = x_642.view(1, 16, 196, -1)
        x_642 = None
        transpose_31 = view_206.transpose(1, 2)
        view_206 = None
        x_643 = transpose_31.reshape(1, 196, -1)
        transpose_31 = None
        x_644 = torch._C._nn.linear(
            x_643,
            l_self_modules_blocks_modules_31_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_31_modules_attn_modules_proj_parameters_bias_,
        )
        x_643 = l_self_modules_blocks_modules_31_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_31_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_645 = torch.nn.functional.dropout(x_644, 0.0, False, False)
        x_644 = None
        x_646 = x_645.view(1, 14, 14, -1)
        x_645 = None
        x_647 = x_639 + x_646
        x_639 = x_646 = None
        x_648 = x_647.reshape(1, 196, -1)
        x_647 = None
        layer_norm_63 = torch.nn.functional.layer_norm(
            x_648,
            (1280,),
            l_self_modules_blocks_modules_31_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_31_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_31_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_31_modules_norm2_parameters_bias_
        ) = None
        x_649 = torch._C._nn.linear(
            layer_norm_63,
            l_self_modules_blocks_modules_31_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_31_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_63 = (
            l_self_modules_blocks_modules_31_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_31_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_650 = torch._C._nn.gelu(x_649, approximate="none")
        x_649 = None
        x_651 = torch.nn.functional.dropout(x_650, 0.0, False, False)
        x_650 = None
        x_652 = torch._C._nn.linear(
            x_651,
            l_self_modules_blocks_modules_31_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_31_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_651 = (
            l_self_modules_blocks_modules_31_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_31_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_653 = torch.nn.functional.dropout(x_652, 0.0, False, False)
        x_652 = None
        x_654 = x_648 + x_653
        x_648 = x_653 = None
        x_655 = x_654.reshape(1, 14, 14, -1)
        x_654 = None
        permute_107 = x_655.permute(0, 3, 1, 2)
        x_655 = None
        input_1 = torch.conv2d(
            permute_107,
            l_self_modules_neck_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        permute_107 = l_self_modules_neck_modules_0_parameters_weight_ = None
        x_656 = input_1.permute(0, 2, 3, 1)
        input_1 = None
        x_657 = torch.nn.functional.layer_norm(
            x_656,
            (256,),
            l_self_modules_neck_modules_1_parameters_weight_,
            l_self_modules_neck_modules_1_parameters_bias_,
            1e-06,
        )
        x_656 = (
            l_self_modules_neck_modules_1_parameters_weight_
        ) = l_self_modules_neck_modules_1_parameters_bias_ = None
        x_658 = x_657.permute(0, 3, 1, 2)
        x_657 = None
        input_2 = torch.conv2d(
            x_658,
            l_self_modules_neck_modules_2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_658 = l_self_modules_neck_modules_2_parameters_weight_ = None
        x_659 = input_2.permute(0, 2, 3, 1)
        input_2 = None
        x_660 = torch.nn.functional.layer_norm(
            x_659,
            (256,),
            l_self_modules_neck_modules_3_parameters_weight_,
            l_self_modules_neck_modules_3_parameters_bias_,
            1e-06,
        )
        x_659 = (
            l_self_modules_neck_modules_3_parameters_weight_
        ) = l_self_modules_neck_modules_3_parameters_bias_ = None
        x_661 = x_660.permute(0, 3, 1, 2)
        x_660 = None
        x_662 = torch.nn.functional.adaptive_avg_pool2d(x_661, 1)
        x_661 = None
        x_663 = x_662.flatten(1, -1)
        x_662 = None
        x_664 = torch.nn.functional.dropout(x_663, 0.0, False, False)
        x_663 = None
        return (x_664,)
