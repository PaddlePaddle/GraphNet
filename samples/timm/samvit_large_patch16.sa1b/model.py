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
        reshape = posemb.reshape(1, 64, 64, 1024)
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
            (1024,),
            l_self_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_0_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_0_modules_norm1_parameters_bias_
        ) = None
        x_5 = torch._C._nn.pad(x_4, (0, 0, 0, 0, 0, 0), "constant", None)
        x_4 = None
        x_6 = x_5.view(1, 1, 14, 1, 14, 1024)
        x_5 = None
        permute_3 = x_6.permute(0, 1, 3, 2, 4, 5)
        x_6 = None
        contiguous = permute_3.contiguous()
        permute_3 = None
        windows = contiguous.view(-1, 14, 14, 1024)
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
        r_q = q.reshape(16, 14, 14, 64)
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
            (1024,),
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
            (1024,),
            l_self_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_1_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_1_modules_norm1_parameters_bias_
        ) = None
        x_26 = torch._C._nn.pad(x_25, (0, 0, 0, 0, 0, 0), "constant", None)
        x_25 = None
        x_27 = x_26.view(1, 1, 14, 1, 14, 1024)
        x_26 = None
        permute_6 = x_27.permute(0, 1, 3, 2, 4, 5)
        x_27 = None
        contiguous_3 = permute_6.contiguous()
        permute_6 = None
        windows_1 = contiguous_3.view(-1, 14, 14, 1024)
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
        r_q_1 = q_1.reshape(16, 14, 14, 64)
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
            (1024,),
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
            (1024,),
            l_self_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_2_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_2_modules_norm1_parameters_bias_
        ) = None
        x_47 = torch._C._nn.pad(x_46, (0, 0, 0, 0, 0, 0), "constant", None)
        x_46 = None
        x_48 = x_47.view(1, 1, 14, 1, 14, 1024)
        x_47 = None
        permute_9 = x_48.permute(0, 1, 3, 2, 4, 5)
        x_48 = None
        contiguous_6 = permute_9.contiguous()
        permute_9 = None
        windows_2 = contiguous_6.view(-1, 14, 14, 1024)
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
        r_q_2 = q_2.reshape(16, 14, 14, 64)
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
            (1024,),
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
            (1024,),
            l_self_modules_blocks_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_3_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_3_modules_norm1_parameters_bias_
        ) = None
        x_68 = torch._C._nn.pad(x_67, (0, 0, 0, 0, 0, 0), "constant", None)
        x_67 = None
        x_69 = x_68.view(1, 1, 14, 1, 14, 1024)
        x_68 = None
        permute_12 = x_69.permute(0, 1, 3, 2, 4, 5)
        x_69 = None
        contiguous_9 = permute_12.contiguous()
        permute_12 = None
        windows_3 = contiguous_9.view(-1, 14, 14, 1024)
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
        r_q_3 = q_3.reshape(16, 14, 14, 64)
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
            (1024,),
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
            (1024,),
            l_self_modules_blocks_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_4_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_4_modules_norm1_parameters_bias_
        ) = None
        x_89 = torch._C._nn.pad(x_88, (0, 0, 0, 0, 0, 0), "constant", None)
        x_88 = None
        x_90 = x_89.view(1, 1, 14, 1, 14, 1024)
        x_89 = None
        permute_15 = x_90.permute(0, 1, 3, 2, 4, 5)
        x_90 = None
        contiguous_12 = permute_15.contiguous()
        permute_15 = None
        windows_4 = contiguous_12.view(-1, 14, 14, 1024)
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
        r_q_4 = q_4.reshape(16, 14, 14, 64)
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
            (1024,),
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
            (1024,),
            l_self_modules_blocks_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_5_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_5_modules_norm1_parameters_bias_
        ) = None
        x_110 = x_109.reshape(1, 196, -1)
        x_109 = None
        linear_20 = torch._C._nn.linear(
            x_110,
            l_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_bias_,
        )
        x_110 = (
            l_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_35 = linear_20.view(1, 196, 3, 16, -1)
        linear_20 = None
        qkv_5 = view_35.permute(2, 0, 3, 1, 4)
        view_35 = None
        reshape_37 = qkv_5.reshape(3, 16, 196, -1)
        qkv_5 = None
        unbind_5 = reshape_37.unbind(0)
        reshape_37 = None
        q_5 = unbind_5[0]
        k_5 = unbind_5[1]
        v_5 = unbind_5[2]
        unbind_5 = None
        reshape_38 = (
            l_self_modules_blocks_modules_5_modules_attn_parameters_rel_pos_h_.reshape(
                1, 127, -1
            )
        )
        l_self_modules_blocks_modules_5_modules_attn_parameters_rel_pos_h_ = None
        permute_19 = reshape_38.permute(0, 2, 1)
        reshape_38 = None
        rel_pos_resized = torch.nn.functional.interpolate(
            permute_19, size=27, mode="linear"
        )
        permute_19 = None
        reshape_39 = rel_pos_resized.reshape(-1, 27)
        rel_pos_resized = None
        rel_pos_resized_1 = reshape_39.permute(1, 0)
        reshape_39 = None
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
        Rh_5 = rel_pos_resized_1[long_10]
        rel_pos_resized_1 = long_10 = None
        reshape_40 = (
            l_self_modules_blocks_modules_5_modules_attn_parameters_rel_pos_w_.reshape(
                1, 127, -1
            )
        )
        l_self_modules_blocks_modules_5_modules_attn_parameters_rel_pos_w_ = None
        permute_21 = reshape_40.permute(0, 2, 1)
        reshape_40 = None
        rel_pos_resized_2 = torch.nn.functional.interpolate(
            permute_21, size=27, mode="linear"
        )
        permute_21 = None
        reshape_41 = rel_pos_resized_2.reshape(-1, 27)
        rel_pos_resized_2 = None
        rel_pos_resized_3 = reshape_41.permute(1, 0)
        reshape_41 = None
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
        Rw_5 = rel_pos_resized_3[long_11]
        rel_pos_resized_3 = long_11 = None
        r_q_5 = q_5.reshape(16, 14, 14, 64)
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
        x_111 = torch._C._nn.scaled_dot_product_attention(
            q_5, k_5, v_5, attn_mask=attn_bias_11, dropout_p=0.0
        )
        q_5 = k_5 = v_5 = attn_bias_11 = None
        view_36 = x_111.view(1, 16, 196, -1)
        x_111 = None
        transpose_5 = view_36.transpose(1, 2)
        view_36 = None
        x_112 = transpose_5.reshape(1, 196, -1)
        transpose_5 = None
        x_113 = torch._C._nn.linear(
            x_112,
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_,
        )
        x_112 = (
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_114 = torch.nn.functional.dropout(x_113, 0.0, False, False)
        x_113 = None
        x_115 = x_114.view(1, 14, 14, -1)
        x_114 = None
        x_116 = x_108 + x_115
        x_108 = x_115 = None
        x_117 = x_116.reshape(1, 196, -1)
        x_116 = None
        layer_norm_11 = torch.nn.functional.layer_norm(
            x_117,
            (1024,),
            l_self_modules_blocks_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_5_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_5_modules_norm2_parameters_bias_
        ) = None
        x_118 = torch._C._nn.linear(
            layer_norm_11,
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_11 = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_119 = torch._C._nn.gelu(x_118, approximate="none")
        x_118 = None
        x_120 = torch.nn.functional.dropout(x_119, 0.0, False, False)
        x_119 = None
        x_121 = torch._C._nn.linear(
            x_120,
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_120 = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_122 = torch.nn.functional.dropout(x_121, 0.0, False, False)
        x_121 = None
        x_123 = x_117 + x_122
        x_117 = x_122 = None
        x_124 = x_123.reshape(1, 14, 14, -1)
        x_123 = None
        x_125 = torch.nn.functional.layer_norm(
            x_124,
            (1024,),
            l_self_modules_blocks_modules_6_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_6_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_6_modules_norm1_parameters_bias_
        ) = None
        x_126 = torch._C._nn.pad(x_125, (0, 0, 0, 0, 0, 0), "constant", None)
        x_125 = None
        x_127 = x_126.view(1, 1, 14, 1, 14, 1024)
        x_126 = None
        permute_23 = x_127.permute(0, 1, 3, 2, 4, 5)
        x_127 = None
        contiguous_15 = permute_23.contiguous()
        permute_23 = None
        windows_5 = contiguous_15.view(-1, 14, 14, 1024)
        contiguous_15 = None
        x_128 = windows_5.reshape(1, 196, -1)
        windows_5 = None
        linear_24 = torch._C._nn.linear(
            x_128,
            l_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_bias_,
        )
        x_128 = (
            l_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_40 = linear_24.view(1, 196, 3, 16, -1)
        linear_24 = None
        qkv_6 = view_40.permute(2, 0, 3, 1, 4)
        view_40 = None
        reshape_48 = qkv_6.reshape(3, 16, 196, -1)
        qkv_6 = None
        unbind_6 = reshape_48.unbind(0)
        reshape_48 = None
        q_6 = unbind_6[0]
        k_6 = unbind_6[1]
        v_6 = unbind_6[2]
        unbind_6 = None
        arange_24 = torch.arange(14)
        getitem_78 = arange_24[(slice(None, None, None), None)]
        arange_24 = None
        q_coords_12 = getitem_78 * 1.0
        getitem_78 = None
        arange_25 = torch.arange(14)
        getitem_79 = arange_25[(None, slice(None, None, None))]
        arange_25 = None
        k_coords_12 = getitem_79 * 1.0
        getitem_79 = None
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
        getitem_81 = arange_26[(slice(None, None, None), None)]
        arange_26 = None
        q_coords_13 = getitem_81 * 1.0
        getitem_81 = None
        arange_27 = torch.arange(14)
        getitem_82 = arange_27[(None, slice(None, None, None))]
        arange_27 = None
        k_coords_13 = getitem_82 * 1.0
        getitem_82 = None
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
        r_q_6 = q_6.reshape(16, 14, 14, 64)
        rel_h_6 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_6, Rh_6)
        Rh_6 = None
        rel_w_6 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_6, Rw_6)
        r_q_6 = Rw_6 = None
        getitem_84 = rel_h_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_6 = None
        getitem_85 = rel_w_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_6 = None
        attn_bias_12 = getitem_84 + getitem_85
        getitem_84 = getitem_85 = None
        attn_bias_13 = attn_bias_12.reshape(-1, 196, 196)
        attn_bias_12 = None
        x_129 = torch._C._nn.scaled_dot_product_attention(
            q_6, k_6, v_6, attn_mask=attn_bias_13, dropout_p=0.0
        )
        q_6 = k_6 = v_6 = attn_bias_13 = None
        view_41 = x_129.view(1, 16, 196, -1)
        x_129 = None
        transpose_6 = view_41.transpose(1, 2)
        view_41 = None
        x_130 = transpose_6.reshape(1, 196, -1)
        transpose_6 = None
        x_131 = torch._C._nn.linear(
            x_130,
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_,
        )
        x_130 = (
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_132 = torch.nn.functional.dropout(x_131, 0.0, False, False)
        x_131 = None
        x_133 = x_132.view(1, 14, 14, -1)
        x_132 = None
        x_134 = x_133.view(1, 1, 1, 14, 14, -1)
        x_133 = None
        permute_25 = x_134.permute(0, 1, 3, 2, 4, 5)
        x_134 = None
        contiguous_16 = permute_25.contiguous()
        permute_25 = None
        x_135 = contiguous_16.view(1, 14, 14, -1)
        contiguous_16 = None
        getitem_86 = x_135[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_135 = None
        x_136 = getitem_86.contiguous()
        getitem_86 = None
        x_137 = x_124 + x_136
        x_124 = x_136 = None
        x_138 = x_137.reshape(1, 196, -1)
        x_137 = None
        layer_norm_13 = torch.nn.functional.layer_norm(
            x_138,
            (1024,),
            l_self_modules_blocks_modules_6_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_6_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_6_modules_norm2_parameters_bias_
        ) = None
        x_139 = torch._C._nn.linear(
            layer_norm_13,
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_13 = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_140 = torch._C._nn.gelu(x_139, approximate="none")
        x_139 = None
        x_141 = torch.nn.functional.dropout(x_140, 0.0, False, False)
        x_140 = None
        x_142 = torch._C._nn.linear(
            x_141,
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_141 = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_143 = torch.nn.functional.dropout(x_142, 0.0, False, False)
        x_142 = None
        x_144 = x_138 + x_143
        x_138 = x_143 = None
        x_145 = x_144.reshape(1, 14, 14, -1)
        x_144 = None
        x_146 = torch.nn.functional.layer_norm(
            x_145,
            (1024,),
            l_self_modules_blocks_modules_7_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_7_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_7_modules_norm1_parameters_bias_
        ) = None
        x_147 = torch._C._nn.pad(x_146, (0, 0, 0, 0, 0, 0), "constant", None)
        x_146 = None
        x_148 = x_147.view(1, 1, 14, 1, 14, 1024)
        x_147 = None
        permute_26 = x_148.permute(0, 1, 3, 2, 4, 5)
        x_148 = None
        contiguous_18 = permute_26.contiguous()
        permute_26 = None
        windows_6 = contiguous_18.view(-1, 14, 14, 1024)
        contiguous_18 = None
        x_149 = windows_6.reshape(1, 196, -1)
        windows_6 = None
        linear_28 = torch._C._nn.linear(
            x_149,
            l_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_bias_,
        )
        x_149 = (
            l_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_47 = linear_28.view(1, 196, 3, 16, -1)
        linear_28 = None
        qkv_7 = view_47.permute(2, 0, 3, 1, 4)
        view_47 = None
        reshape_55 = qkv_7.reshape(3, 16, 196, -1)
        qkv_7 = None
        unbind_7 = reshape_55.unbind(0)
        reshape_55 = None
        q_7 = unbind_7[0]
        k_7 = unbind_7[1]
        v_7 = unbind_7[2]
        unbind_7 = None
        arange_28 = torch.arange(14)
        getitem_90 = arange_28[(slice(None, None, None), None)]
        arange_28 = None
        q_coords_14 = getitem_90 * 1.0
        getitem_90 = None
        arange_29 = torch.arange(14)
        getitem_91 = arange_29[(None, slice(None, None, None))]
        arange_29 = None
        k_coords_14 = getitem_91 * 1.0
        getitem_91 = None
        sub_14 = q_coords_14 - k_coords_14
        q_coords_14 = k_coords_14 = None
        relative_coords_14 = sub_14 + 13.0
        sub_14 = None
        long_14 = relative_coords_14.long()
        relative_coords_14 = None
        Rh_7 = l_self_modules_blocks_modules_7_modules_attn_parameters_rel_pos_h_[
            long_14
        ]
        l_self_modules_blocks_modules_7_modules_attn_parameters_rel_pos_h_ = (
            long_14
        ) = None
        arange_30 = torch.arange(14)
        getitem_93 = arange_30[(slice(None, None, None), None)]
        arange_30 = None
        q_coords_15 = getitem_93 * 1.0
        getitem_93 = None
        arange_31 = torch.arange(14)
        getitem_94 = arange_31[(None, slice(None, None, None))]
        arange_31 = None
        k_coords_15 = getitem_94 * 1.0
        getitem_94 = None
        sub_15 = q_coords_15 - k_coords_15
        q_coords_15 = k_coords_15 = None
        relative_coords_15 = sub_15 + 13.0
        sub_15 = None
        long_15 = relative_coords_15.long()
        relative_coords_15 = None
        Rw_7 = l_self_modules_blocks_modules_7_modules_attn_parameters_rel_pos_w_[
            long_15
        ]
        l_self_modules_blocks_modules_7_modules_attn_parameters_rel_pos_w_ = (
            long_15
        ) = None
        r_q_7 = q_7.reshape(16, 14, 14, 64)
        rel_h_7 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_7, Rh_7)
        Rh_7 = None
        rel_w_7 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_7, Rw_7)
        r_q_7 = Rw_7 = None
        getitem_96 = rel_h_7[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_7 = None
        getitem_97 = rel_w_7[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_7 = None
        attn_bias_14 = getitem_96 + getitem_97
        getitem_96 = getitem_97 = None
        attn_bias_15 = attn_bias_14.reshape(-1, 196, 196)
        attn_bias_14 = None
        x_150 = torch._C._nn.scaled_dot_product_attention(
            q_7, k_7, v_7, attn_mask=attn_bias_15, dropout_p=0.0
        )
        q_7 = k_7 = v_7 = attn_bias_15 = None
        view_48 = x_150.view(1, 16, 196, -1)
        x_150 = None
        transpose_7 = view_48.transpose(1, 2)
        view_48 = None
        x_151 = transpose_7.reshape(1, 196, -1)
        transpose_7 = None
        x_152 = torch._C._nn.linear(
            x_151,
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_,
        )
        x_151 = (
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_153 = torch.nn.functional.dropout(x_152, 0.0, False, False)
        x_152 = None
        x_154 = x_153.view(1, 14, 14, -1)
        x_153 = None
        x_155 = x_154.view(1, 1, 1, 14, 14, -1)
        x_154 = None
        permute_28 = x_155.permute(0, 1, 3, 2, 4, 5)
        x_155 = None
        contiguous_19 = permute_28.contiguous()
        permute_28 = None
        x_156 = contiguous_19.view(1, 14, 14, -1)
        contiguous_19 = None
        getitem_98 = x_156[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_156 = None
        x_157 = getitem_98.contiguous()
        getitem_98 = None
        x_158 = x_145 + x_157
        x_145 = x_157 = None
        x_159 = x_158.reshape(1, 196, -1)
        x_158 = None
        layer_norm_15 = torch.nn.functional.layer_norm(
            x_159,
            (1024,),
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
            (1024,),
            l_self_modules_blocks_modules_8_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_8_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_8_modules_norm1_parameters_bias_
        ) = None
        x_168 = torch._C._nn.pad(x_167, (0, 0, 0, 0, 0, 0), "constant", None)
        x_167 = None
        x_169 = x_168.view(1, 1, 14, 1, 14, 1024)
        x_168 = None
        permute_29 = x_169.permute(0, 1, 3, 2, 4, 5)
        x_169 = None
        contiguous_21 = permute_29.contiguous()
        permute_29 = None
        windows_7 = contiguous_21.view(-1, 14, 14, 1024)
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
        r_q_8 = q_8.reshape(16, 14, 14, 64)
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
            (1024,),
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
            (1024,),
            l_self_modules_blocks_modules_9_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_9_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_9_modules_norm1_parameters_bias_
        ) = None
        x_189 = torch._C._nn.pad(x_188, (0, 0, 0, 0, 0, 0), "constant", None)
        x_188 = None
        x_190 = x_189.view(1, 1, 14, 1, 14, 1024)
        x_189 = None
        permute_32 = x_190.permute(0, 1, 3, 2, 4, 5)
        x_190 = None
        contiguous_24 = permute_32.contiguous()
        permute_32 = None
        windows_8 = contiguous_24.view(-1, 14, 14, 1024)
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
        r_q_9 = q_9.reshape(16, 14, 14, 64)
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
            (1024,),
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
            (1024,),
            l_self_modules_blocks_modules_10_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_10_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_10_modules_norm1_parameters_bias_
        ) = None
        x_210 = torch._C._nn.pad(x_209, (0, 0, 0, 0, 0, 0), "constant", None)
        x_209 = None
        x_211 = x_210.view(1, 1, 14, 1, 14, 1024)
        x_210 = None
        permute_35 = x_211.permute(0, 1, 3, 2, 4, 5)
        x_211 = None
        contiguous_27 = permute_35.contiguous()
        permute_35 = None
        windows_9 = contiguous_27.view(-1, 14, 14, 1024)
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
        r_q_10 = q_10.reshape(16, 14, 14, 64)
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
            (1024,),
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
            (1024,),
            l_self_modules_blocks_modules_11_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_11_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_11_modules_norm1_parameters_bias_
        ) = None
        x_231 = x_230.reshape(1, 196, -1)
        x_230 = None
        linear_44 = torch._C._nn.linear(
            x_231,
            l_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_bias_,
        )
        x_231 = (
            l_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_73 = linear_44.view(1, 196, 3, 16, -1)
        linear_44 = None
        qkv_11 = view_73.permute(2, 0, 3, 1, 4)
        view_73 = None
        reshape_83 = qkv_11.reshape(3, 16, 196, -1)
        qkv_11 = None
        unbind_11 = reshape_83.unbind(0)
        reshape_83 = None
        q_11 = unbind_11[0]
        k_11 = unbind_11[1]
        v_11 = unbind_11[2]
        unbind_11 = None
        reshape_84 = (
            l_self_modules_blocks_modules_11_modules_attn_parameters_rel_pos_h_.reshape(
                1, 127, -1
            )
        )
        l_self_modules_blocks_modules_11_modules_attn_parameters_rel_pos_h_ = None
        permute_39 = reshape_84.permute(0, 2, 1)
        reshape_84 = None
        rel_pos_resized_4 = torch.nn.functional.interpolate(
            permute_39, size=27, mode="linear"
        )
        permute_39 = None
        reshape_85 = rel_pos_resized_4.reshape(-1, 27)
        rel_pos_resized_4 = None
        rel_pos_resized_5 = reshape_85.permute(1, 0)
        reshape_85 = None
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
        Rh_11 = rel_pos_resized_5[long_22]
        rel_pos_resized_5 = long_22 = None
        reshape_86 = (
            l_self_modules_blocks_modules_11_modules_attn_parameters_rel_pos_w_.reshape(
                1, 127, -1
            )
        )
        l_self_modules_blocks_modules_11_modules_attn_parameters_rel_pos_w_ = None
        permute_41 = reshape_86.permute(0, 2, 1)
        reshape_86 = None
        rel_pos_resized_6 = torch.nn.functional.interpolate(
            permute_41, size=27, mode="linear"
        )
        permute_41 = None
        reshape_87 = rel_pos_resized_6.reshape(-1, 27)
        rel_pos_resized_6 = None
        rel_pos_resized_7 = reshape_87.permute(1, 0)
        reshape_87 = None
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
        Rw_11 = rel_pos_resized_7[long_23]
        rel_pos_resized_7 = long_23 = None
        r_q_11 = q_11.reshape(16, 14, 14, 64)
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
        x_232 = torch._C._nn.scaled_dot_product_attention(
            q_11, k_11, v_11, attn_mask=attn_bias_23, dropout_p=0.0
        )
        q_11 = k_11 = v_11 = attn_bias_23 = None
        view_74 = x_232.view(1, 16, 196, -1)
        x_232 = None
        transpose_11 = view_74.transpose(1, 2)
        view_74 = None
        x_233 = transpose_11.reshape(1, 196, -1)
        transpose_11 = None
        x_234 = torch._C._nn.linear(
            x_233,
            l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_,
        )
        x_233 = l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_235 = torch.nn.functional.dropout(x_234, 0.0, False, False)
        x_234 = None
        x_236 = x_235.view(1, 14, 14, -1)
        x_235 = None
        x_237 = x_229 + x_236
        x_229 = x_236 = None
        x_238 = x_237.reshape(1, 196, -1)
        x_237 = None
        layer_norm_23 = torch.nn.functional.layer_norm(
            x_238,
            (1024,),
            l_self_modules_blocks_modules_11_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_11_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_11_modules_norm2_parameters_bias_
        ) = None
        x_239 = torch._C._nn.linear(
            layer_norm_23,
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_23 = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_240 = torch._C._nn.gelu(x_239, approximate="none")
        x_239 = None
        x_241 = torch.nn.functional.dropout(x_240, 0.0, False, False)
        x_240 = None
        x_242 = torch._C._nn.linear(
            x_241,
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_241 = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_243 = torch.nn.functional.dropout(x_242, 0.0, False, False)
        x_242 = None
        x_244 = x_238 + x_243
        x_238 = x_243 = None
        x_245 = x_244.reshape(1, 14, 14, -1)
        x_244 = None
        x_246 = torch.nn.functional.layer_norm(
            x_245,
            (1024,),
            l_self_modules_blocks_modules_12_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_12_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_12_modules_norm1_parameters_bias_
        ) = None
        x_247 = torch._C._nn.pad(x_246, (0, 0, 0, 0, 0, 0), "constant", None)
        x_246 = None
        x_248 = x_247.view(1, 1, 14, 1, 14, 1024)
        x_247 = None
        permute_43 = x_248.permute(0, 1, 3, 2, 4, 5)
        x_248 = None
        contiguous_30 = permute_43.contiguous()
        permute_43 = None
        windows_10 = contiguous_30.view(-1, 14, 14, 1024)
        contiguous_30 = None
        x_249 = windows_10.reshape(1, 196, -1)
        windows_10 = None
        linear_48 = torch._C._nn.linear(
            x_249,
            l_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_bias_,
        )
        x_249 = (
            l_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_78 = linear_48.view(1, 196, 3, 16, -1)
        linear_48 = None
        qkv_12 = view_78.permute(2, 0, 3, 1, 4)
        view_78 = None
        reshape_94 = qkv_12.reshape(3, 16, 196, -1)
        qkv_12 = None
        unbind_12 = reshape_94.unbind(0)
        reshape_94 = None
        q_12 = unbind_12[0]
        k_12 = unbind_12[1]
        v_12 = unbind_12[2]
        unbind_12 = None
        arange_48 = torch.arange(14)
        getitem_149 = arange_48[(slice(None, None, None), None)]
        arange_48 = None
        q_coords_24 = getitem_149 * 1.0
        getitem_149 = None
        arange_49 = torch.arange(14)
        getitem_150 = arange_49[(None, slice(None, None, None))]
        arange_49 = None
        k_coords_24 = getitem_150 * 1.0
        getitem_150 = None
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
        getitem_152 = arange_50[(slice(None, None, None), None)]
        arange_50 = None
        q_coords_25 = getitem_152 * 1.0
        getitem_152 = None
        arange_51 = torch.arange(14)
        getitem_153 = arange_51[(None, slice(None, None, None))]
        arange_51 = None
        k_coords_25 = getitem_153 * 1.0
        getitem_153 = None
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
        r_q_12 = q_12.reshape(16, 14, 14, 64)
        rel_h_12 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_12, Rh_12)
        Rh_12 = None
        rel_w_12 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_12, Rw_12)
        r_q_12 = Rw_12 = None
        getitem_155 = rel_h_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_12 = None
        getitem_156 = rel_w_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_12 = None
        attn_bias_24 = getitem_155 + getitem_156
        getitem_155 = getitem_156 = None
        attn_bias_25 = attn_bias_24.reshape(-1, 196, 196)
        attn_bias_24 = None
        x_250 = torch._C._nn.scaled_dot_product_attention(
            q_12, k_12, v_12, attn_mask=attn_bias_25, dropout_p=0.0
        )
        q_12 = k_12 = v_12 = attn_bias_25 = None
        view_79 = x_250.view(1, 16, 196, -1)
        x_250 = None
        transpose_12 = view_79.transpose(1, 2)
        view_79 = None
        x_251 = transpose_12.reshape(1, 196, -1)
        transpose_12 = None
        x_252 = torch._C._nn.linear(
            x_251,
            l_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_,
        )
        x_251 = l_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_253 = torch.nn.functional.dropout(x_252, 0.0, False, False)
        x_252 = None
        x_254 = x_253.view(1, 14, 14, -1)
        x_253 = None
        x_255 = x_254.view(1, 1, 1, 14, 14, -1)
        x_254 = None
        permute_45 = x_255.permute(0, 1, 3, 2, 4, 5)
        x_255 = None
        contiguous_31 = permute_45.contiguous()
        permute_45 = None
        x_256 = contiguous_31.view(1, 14, 14, -1)
        contiguous_31 = None
        getitem_157 = x_256[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_256 = None
        x_257 = getitem_157.contiguous()
        getitem_157 = None
        x_258 = x_245 + x_257
        x_245 = x_257 = None
        x_259 = x_258.reshape(1, 196, -1)
        x_258 = None
        layer_norm_25 = torch.nn.functional.layer_norm(
            x_259,
            (1024,),
            l_self_modules_blocks_modules_12_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_12_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_12_modules_norm2_parameters_bias_
        ) = None
        x_260 = torch._C._nn.linear(
            layer_norm_25,
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_25 = (
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_261 = torch._C._nn.gelu(x_260, approximate="none")
        x_260 = None
        x_262 = torch.nn.functional.dropout(x_261, 0.0, False, False)
        x_261 = None
        x_263 = torch._C._nn.linear(
            x_262,
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_262 = (
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_264 = torch.nn.functional.dropout(x_263, 0.0, False, False)
        x_263 = None
        x_265 = x_259 + x_264
        x_259 = x_264 = None
        x_266 = x_265.reshape(1, 14, 14, -1)
        x_265 = None
        x_267 = torch.nn.functional.layer_norm(
            x_266,
            (1024,),
            l_self_modules_blocks_modules_13_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_13_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_13_modules_norm1_parameters_bias_
        ) = None
        x_268 = torch._C._nn.pad(x_267, (0, 0, 0, 0, 0, 0), "constant", None)
        x_267 = None
        x_269 = x_268.view(1, 1, 14, 1, 14, 1024)
        x_268 = None
        permute_46 = x_269.permute(0, 1, 3, 2, 4, 5)
        x_269 = None
        contiguous_33 = permute_46.contiguous()
        permute_46 = None
        windows_11 = contiguous_33.view(-1, 14, 14, 1024)
        contiguous_33 = None
        x_270 = windows_11.reshape(1, 196, -1)
        windows_11 = None
        linear_52 = torch._C._nn.linear(
            x_270,
            l_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_bias_,
        )
        x_270 = (
            l_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_85 = linear_52.view(1, 196, 3, 16, -1)
        linear_52 = None
        qkv_13 = view_85.permute(2, 0, 3, 1, 4)
        view_85 = None
        reshape_101 = qkv_13.reshape(3, 16, 196, -1)
        qkv_13 = None
        unbind_13 = reshape_101.unbind(0)
        reshape_101 = None
        q_13 = unbind_13[0]
        k_13 = unbind_13[1]
        v_13 = unbind_13[2]
        unbind_13 = None
        arange_52 = torch.arange(14)
        getitem_161 = arange_52[(slice(None, None, None), None)]
        arange_52 = None
        q_coords_26 = getitem_161 * 1.0
        getitem_161 = None
        arange_53 = torch.arange(14)
        getitem_162 = arange_53[(None, slice(None, None, None))]
        arange_53 = None
        k_coords_26 = getitem_162 * 1.0
        getitem_162 = None
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
        getitem_164 = arange_54[(slice(None, None, None), None)]
        arange_54 = None
        q_coords_27 = getitem_164 * 1.0
        getitem_164 = None
        arange_55 = torch.arange(14)
        getitem_165 = arange_55[(None, slice(None, None, None))]
        arange_55 = None
        k_coords_27 = getitem_165 * 1.0
        getitem_165 = None
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
        r_q_13 = q_13.reshape(16, 14, 14, 64)
        rel_h_13 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_13, Rh_13)
        Rh_13 = None
        rel_w_13 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_13, Rw_13)
        r_q_13 = Rw_13 = None
        getitem_167 = rel_h_13[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_13 = None
        getitem_168 = rel_w_13[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_13 = None
        attn_bias_26 = getitem_167 + getitem_168
        getitem_167 = getitem_168 = None
        attn_bias_27 = attn_bias_26.reshape(-1, 196, 196)
        attn_bias_26 = None
        x_271 = torch._C._nn.scaled_dot_product_attention(
            q_13, k_13, v_13, attn_mask=attn_bias_27, dropout_p=0.0
        )
        q_13 = k_13 = v_13 = attn_bias_27 = None
        view_86 = x_271.view(1, 16, 196, -1)
        x_271 = None
        transpose_13 = view_86.transpose(1, 2)
        view_86 = None
        x_272 = transpose_13.reshape(1, 196, -1)
        transpose_13 = None
        x_273 = torch._C._nn.linear(
            x_272,
            l_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_,
        )
        x_272 = l_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_274 = torch.nn.functional.dropout(x_273, 0.0, False, False)
        x_273 = None
        x_275 = x_274.view(1, 14, 14, -1)
        x_274 = None
        x_276 = x_275.view(1, 1, 1, 14, 14, -1)
        x_275 = None
        permute_48 = x_276.permute(0, 1, 3, 2, 4, 5)
        x_276 = None
        contiguous_34 = permute_48.contiguous()
        permute_48 = None
        x_277 = contiguous_34.view(1, 14, 14, -1)
        contiguous_34 = None
        getitem_169 = x_277[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_277 = None
        x_278 = getitem_169.contiguous()
        getitem_169 = None
        x_279 = x_266 + x_278
        x_266 = x_278 = None
        x_280 = x_279.reshape(1, 196, -1)
        x_279 = None
        layer_norm_27 = torch.nn.functional.layer_norm(
            x_280,
            (1024,),
            l_self_modules_blocks_modules_13_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_13_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_13_modules_norm2_parameters_bias_
        ) = None
        x_281 = torch._C._nn.linear(
            layer_norm_27,
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_27 = (
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_282 = torch._C._nn.gelu(x_281, approximate="none")
        x_281 = None
        x_283 = torch.nn.functional.dropout(x_282, 0.0, False, False)
        x_282 = None
        x_284 = torch._C._nn.linear(
            x_283,
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_283 = (
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_285 = torch.nn.functional.dropout(x_284, 0.0, False, False)
        x_284 = None
        x_286 = x_280 + x_285
        x_280 = x_285 = None
        x_287 = x_286.reshape(1, 14, 14, -1)
        x_286 = None
        x_288 = torch.nn.functional.layer_norm(
            x_287,
            (1024,),
            l_self_modules_blocks_modules_14_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_14_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_14_modules_norm1_parameters_bias_
        ) = None
        x_289 = torch._C._nn.pad(x_288, (0, 0, 0, 0, 0, 0), "constant", None)
        x_288 = None
        x_290 = x_289.view(1, 1, 14, 1, 14, 1024)
        x_289 = None
        permute_49 = x_290.permute(0, 1, 3, 2, 4, 5)
        x_290 = None
        contiguous_36 = permute_49.contiguous()
        permute_49 = None
        windows_12 = contiguous_36.view(-1, 14, 14, 1024)
        contiguous_36 = None
        x_291 = windows_12.reshape(1, 196, -1)
        windows_12 = None
        linear_56 = torch._C._nn.linear(
            x_291,
            l_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_bias_,
        )
        x_291 = (
            l_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_92 = linear_56.view(1, 196, 3, 16, -1)
        linear_56 = None
        qkv_14 = view_92.permute(2, 0, 3, 1, 4)
        view_92 = None
        reshape_108 = qkv_14.reshape(3, 16, 196, -1)
        qkv_14 = None
        unbind_14 = reshape_108.unbind(0)
        reshape_108 = None
        q_14 = unbind_14[0]
        k_14 = unbind_14[1]
        v_14 = unbind_14[2]
        unbind_14 = None
        arange_56 = torch.arange(14)
        getitem_173 = arange_56[(slice(None, None, None), None)]
        arange_56 = None
        q_coords_28 = getitem_173 * 1.0
        getitem_173 = None
        arange_57 = torch.arange(14)
        getitem_174 = arange_57[(None, slice(None, None, None))]
        arange_57 = None
        k_coords_28 = getitem_174 * 1.0
        getitem_174 = None
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
        getitem_176 = arange_58[(slice(None, None, None), None)]
        arange_58 = None
        q_coords_29 = getitem_176 * 1.0
        getitem_176 = None
        arange_59 = torch.arange(14)
        getitem_177 = arange_59[(None, slice(None, None, None))]
        arange_59 = None
        k_coords_29 = getitem_177 * 1.0
        getitem_177 = None
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
        r_q_14 = q_14.reshape(16, 14, 14, 64)
        rel_h_14 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_14, Rh_14)
        Rh_14 = None
        rel_w_14 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_14, Rw_14)
        r_q_14 = Rw_14 = None
        getitem_179 = rel_h_14[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_14 = None
        getitem_180 = rel_w_14[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_14 = None
        attn_bias_28 = getitem_179 + getitem_180
        getitem_179 = getitem_180 = None
        attn_bias_29 = attn_bias_28.reshape(-1, 196, 196)
        attn_bias_28 = None
        x_292 = torch._C._nn.scaled_dot_product_attention(
            q_14, k_14, v_14, attn_mask=attn_bias_29, dropout_p=0.0
        )
        q_14 = k_14 = v_14 = attn_bias_29 = None
        view_93 = x_292.view(1, 16, 196, -1)
        x_292 = None
        transpose_14 = view_93.transpose(1, 2)
        view_93 = None
        x_293 = transpose_14.reshape(1, 196, -1)
        transpose_14 = None
        x_294 = torch._C._nn.linear(
            x_293,
            l_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_bias_,
        )
        x_293 = l_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_295 = torch.nn.functional.dropout(x_294, 0.0, False, False)
        x_294 = None
        x_296 = x_295.view(1, 14, 14, -1)
        x_295 = None
        x_297 = x_296.view(1, 1, 1, 14, 14, -1)
        x_296 = None
        permute_51 = x_297.permute(0, 1, 3, 2, 4, 5)
        x_297 = None
        contiguous_37 = permute_51.contiguous()
        permute_51 = None
        x_298 = contiguous_37.view(1, 14, 14, -1)
        contiguous_37 = None
        getitem_181 = x_298[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_298 = None
        x_299 = getitem_181.contiguous()
        getitem_181 = None
        x_300 = x_287 + x_299
        x_287 = x_299 = None
        x_301 = x_300.reshape(1, 196, -1)
        x_300 = None
        layer_norm_29 = torch.nn.functional.layer_norm(
            x_301,
            (1024,),
            l_self_modules_blocks_modules_14_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_14_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_14_modules_norm2_parameters_bias_
        ) = None
        x_302 = torch._C._nn.linear(
            layer_norm_29,
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_29 = (
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_303 = torch._C._nn.gelu(x_302, approximate="none")
        x_302 = None
        x_304 = torch.nn.functional.dropout(x_303, 0.0, False, False)
        x_303 = None
        x_305 = torch._C._nn.linear(
            x_304,
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_304 = (
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_306 = torch.nn.functional.dropout(x_305, 0.0, False, False)
        x_305 = None
        x_307 = x_301 + x_306
        x_301 = x_306 = None
        x_308 = x_307.reshape(1, 14, 14, -1)
        x_307 = None
        x_309 = torch.nn.functional.layer_norm(
            x_308,
            (1024,),
            l_self_modules_blocks_modules_15_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_15_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_15_modules_norm1_parameters_bias_
        ) = None
        x_310 = torch._C._nn.pad(x_309, (0, 0, 0, 0, 0, 0), "constant", None)
        x_309 = None
        x_311 = x_310.view(1, 1, 14, 1, 14, 1024)
        x_310 = None
        permute_52 = x_311.permute(0, 1, 3, 2, 4, 5)
        x_311 = None
        contiguous_39 = permute_52.contiguous()
        permute_52 = None
        windows_13 = contiguous_39.view(-1, 14, 14, 1024)
        contiguous_39 = None
        x_312 = windows_13.reshape(1, 196, -1)
        windows_13 = None
        linear_60 = torch._C._nn.linear(
            x_312,
            l_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_bias_,
        )
        x_312 = (
            l_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_99 = linear_60.view(1, 196, 3, 16, -1)
        linear_60 = None
        qkv_15 = view_99.permute(2, 0, 3, 1, 4)
        view_99 = None
        reshape_115 = qkv_15.reshape(3, 16, 196, -1)
        qkv_15 = None
        unbind_15 = reshape_115.unbind(0)
        reshape_115 = None
        q_15 = unbind_15[0]
        k_15 = unbind_15[1]
        v_15 = unbind_15[2]
        unbind_15 = None
        arange_60 = torch.arange(14)
        getitem_185 = arange_60[(slice(None, None, None), None)]
        arange_60 = None
        q_coords_30 = getitem_185 * 1.0
        getitem_185 = None
        arange_61 = torch.arange(14)
        getitem_186 = arange_61[(None, slice(None, None, None))]
        arange_61 = None
        k_coords_30 = getitem_186 * 1.0
        getitem_186 = None
        sub_30 = q_coords_30 - k_coords_30
        q_coords_30 = k_coords_30 = None
        relative_coords_30 = sub_30 + 13.0
        sub_30 = None
        long_30 = relative_coords_30.long()
        relative_coords_30 = None
        Rh_15 = l_self_modules_blocks_modules_15_modules_attn_parameters_rel_pos_h_[
            long_30
        ]
        l_self_modules_blocks_modules_15_modules_attn_parameters_rel_pos_h_ = (
            long_30
        ) = None
        arange_62 = torch.arange(14)
        getitem_188 = arange_62[(slice(None, None, None), None)]
        arange_62 = None
        q_coords_31 = getitem_188 * 1.0
        getitem_188 = None
        arange_63 = torch.arange(14)
        getitem_189 = arange_63[(None, slice(None, None, None))]
        arange_63 = None
        k_coords_31 = getitem_189 * 1.0
        getitem_189 = None
        sub_31 = q_coords_31 - k_coords_31
        q_coords_31 = k_coords_31 = None
        relative_coords_31 = sub_31 + 13.0
        sub_31 = None
        long_31 = relative_coords_31.long()
        relative_coords_31 = None
        Rw_15 = l_self_modules_blocks_modules_15_modules_attn_parameters_rel_pos_w_[
            long_31
        ]
        l_self_modules_blocks_modules_15_modules_attn_parameters_rel_pos_w_ = (
            long_31
        ) = None
        r_q_15 = q_15.reshape(16, 14, 14, 64)
        rel_h_15 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_15, Rh_15)
        Rh_15 = None
        rel_w_15 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_15, Rw_15)
        r_q_15 = Rw_15 = None
        getitem_191 = rel_h_15[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_15 = None
        getitem_192 = rel_w_15[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_15 = None
        attn_bias_30 = getitem_191 + getitem_192
        getitem_191 = getitem_192 = None
        attn_bias_31 = attn_bias_30.reshape(-1, 196, 196)
        attn_bias_30 = None
        x_313 = torch._C._nn.scaled_dot_product_attention(
            q_15, k_15, v_15, attn_mask=attn_bias_31, dropout_p=0.0
        )
        q_15 = k_15 = v_15 = attn_bias_31 = None
        view_100 = x_313.view(1, 16, 196, -1)
        x_313 = None
        transpose_15 = view_100.transpose(1, 2)
        view_100 = None
        x_314 = transpose_15.reshape(1, 196, -1)
        transpose_15 = None
        x_315 = torch._C._nn.linear(
            x_314,
            l_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_bias_,
        )
        x_314 = l_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_316 = torch.nn.functional.dropout(x_315, 0.0, False, False)
        x_315 = None
        x_317 = x_316.view(1, 14, 14, -1)
        x_316 = None
        x_318 = x_317.view(1, 1, 1, 14, 14, -1)
        x_317 = None
        permute_54 = x_318.permute(0, 1, 3, 2, 4, 5)
        x_318 = None
        contiguous_40 = permute_54.contiguous()
        permute_54 = None
        x_319 = contiguous_40.view(1, 14, 14, -1)
        contiguous_40 = None
        getitem_193 = x_319[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_319 = None
        x_320 = getitem_193.contiguous()
        getitem_193 = None
        x_321 = x_308 + x_320
        x_308 = x_320 = None
        x_322 = x_321.reshape(1, 196, -1)
        x_321 = None
        layer_norm_31 = torch.nn.functional.layer_norm(
            x_322,
            (1024,),
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
            (1024,),
            l_self_modules_blocks_modules_16_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_16_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_16_modules_norm1_parameters_bias_
        ) = None
        x_331 = torch._C._nn.pad(x_330, (0, 0, 0, 0, 0, 0), "constant", None)
        x_330 = None
        x_332 = x_331.view(1, 1, 14, 1, 14, 1024)
        x_331 = None
        permute_55 = x_332.permute(0, 1, 3, 2, 4, 5)
        x_332 = None
        contiguous_42 = permute_55.contiguous()
        permute_55 = None
        windows_14 = contiguous_42.view(-1, 14, 14, 1024)
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
        r_q_16 = q_16.reshape(16, 14, 14, 64)
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
            (1024,),
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
            (1024,),
            l_self_modules_blocks_modules_17_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_17_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_17_modules_norm1_parameters_bias_
        ) = None
        x_352 = x_351.reshape(1, 196, -1)
        x_351 = None
        linear_68 = torch._C._nn.linear(
            x_352,
            l_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_bias_,
        )
        x_352 = (
            l_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_111 = linear_68.view(1, 196, 3, 16, -1)
        linear_68 = None
        qkv_17 = view_111.permute(2, 0, 3, 1, 4)
        view_111 = None
        reshape_129 = qkv_17.reshape(3, 16, 196, -1)
        qkv_17 = None
        unbind_17 = reshape_129.unbind(0)
        reshape_129 = None
        q_17 = unbind_17[0]
        k_17 = unbind_17[1]
        v_17 = unbind_17[2]
        unbind_17 = None
        reshape_130 = (
            l_self_modules_blocks_modules_17_modules_attn_parameters_rel_pos_h_.reshape(
                1, 127, -1
            )
        )
        l_self_modules_blocks_modules_17_modules_attn_parameters_rel_pos_h_ = None
        permute_59 = reshape_130.permute(0, 2, 1)
        reshape_130 = None
        rel_pos_resized_8 = torch.nn.functional.interpolate(
            permute_59, size=27, mode="linear"
        )
        permute_59 = None
        reshape_131 = rel_pos_resized_8.reshape(-1, 27)
        rel_pos_resized_8 = None
        rel_pos_resized_9 = reshape_131.permute(1, 0)
        reshape_131 = None
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
        Rh_17 = rel_pos_resized_9[long_34]
        rel_pos_resized_9 = long_34 = None
        reshape_132 = (
            l_self_modules_blocks_modules_17_modules_attn_parameters_rel_pos_w_.reshape(
                1, 127, -1
            )
        )
        l_self_modules_blocks_modules_17_modules_attn_parameters_rel_pos_w_ = None
        permute_61 = reshape_132.permute(0, 2, 1)
        reshape_132 = None
        rel_pos_resized_10 = torch.nn.functional.interpolate(
            permute_61, size=27, mode="linear"
        )
        permute_61 = None
        reshape_133 = rel_pos_resized_10.reshape(-1, 27)
        rel_pos_resized_10 = None
        rel_pos_resized_11 = reshape_133.permute(1, 0)
        reshape_133 = None
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
        Rw_17 = rel_pos_resized_11[long_35]
        rel_pos_resized_11 = long_35 = None
        r_q_17 = q_17.reshape(16, 14, 14, 64)
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
        x_353 = torch._C._nn.scaled_dot_product_attention(
            q_17, k_17, v_17, attn_mask=attn_bias_35, dropout_p=0.0
        )
        q_17 = k_17 = v_17 = attn_bias_35 = None
        view_112 = x_353.view(1, 16, 196, -1)
        x_353 = None
        transpose_17 = view_112.transpose(1, 2)
        view_112 = None
        x_354 = transpose_17.reshape(1, 196, -1)
        transpose_17 = None
        x_355 = torch._C._nn.linear(
            x_354,
            l_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_bias_,
        )
        x_354 = l_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_356 = torch.nn.functional.dropout(x_355, 0.0, False, False)
        x_355 = None
        x_357 = x_356.view(1, 14, 14, -1)
        x_356 = None
        x_358 = x_350 + x_357
        x_350 = x_357 = None
        x_359 = x_358.reshape(1, 196, -1)
        x_358 = None
        layer_norm_35 = torch.nn.functional.layer_norm(
            x_359,
            (1024,),
            l_self_modules_blocks_modules_17_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_17_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_17_modules_norm2_parameters_bias_
        ) = None
        x_360 = torch._C._nn.linear(
            layer_norm_35,
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_35 = (
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_361 = torch._C._nn.gelu(x_360, approximate="none")
        x_360 = None
        x_362 = torch.nn.functional.dropout(x_361, 0.0, False, False)
        x_361 = None
        x_363 = torch._C._nn.linear(
            x_362,
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_362 = (
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_364 = torch.nn.functional.dropout(x_363, 0.0, False, False)
        x_363 = None
        x_365 = x_359 + x_364
        x_359 = x_364 = None
        x_366 = x_365.reshape(1, 14, 14, -1)
        x_365 = None
        x_367 = torch.nn.functional.layer_norm(
            x_366,
            (1024,),
            l_self_modules_blocks_modules_18_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_18_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_18_modules_norm1_parameters_bias_
        ) = None
        x_368 = torch._C._nn.pad(x_367, (0, 0, 0, 0, 0, 0), "constant", None)
        x_367 = None
        x_369 = x_368.view(1, 1, 14, 1, 14, 1024)
        x_368 = None
        permute_63 = x_369.permute(0, 1, 3, 2, 4, 5)
        x_369 = None
        contiguous_45 = permute_63.contiguous()
        permute_63 = None
        windows_15 = contiguous_45.view(-1, 14, 14, 1024)
        contiguous_45 = None
        x_370 = windows_15.reshape(1, 196, -1)
        windows_15 = None
        linear_72 = torch._C._nn.linear(
            x_370,
            l_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_bias_,
        )
        x_370 = (
            l_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_116 = linear_72.view(1, 196, 3, 16, -1)
        linear_72 = None
        qkv_18 = view_116.permute(2, 0, 3, 1, 4)
        view_116 = None
        reshape_140 = qkv_18.reshape(3, 16, 196, -1)
        qkv_18 = None
        unbind_18 = reshape_140.unbind(0)
        reshape_140 = None
        q_18 = unbind_18[0]
        k_18 = unbind_18[1]
        v_18 = unbind_18[2]
        unbind_18 = None
        arange_72 = torch.arange(14)
        getitem_220 = arange_72[(slice(None, None, None), None)]
        arange_72 = None
        q_coords_36 = getitem_220 * 1.0
        getitem_220 = None
        arange_73 = torch.arange(14)
        getitem_221 = arange_73[(None, slice(None, None, None))]
        arange_73 = None
        k_coords_36 = getitem_221 * 1.0
        getitem_221 = None
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
        getitem_223 = arange_74[(slice(None, None, None), None)]
        arange_74 = None
        q_coords_37 = getitem_223 * 1.0
        getitem_223 = None
        arange_75 = torch.arange(14)
        getitem_224 = arange_75[(None, slice(None, None, None))]
        arange_75 = None
        k_coords_37 = getitem_224 * 1.0
        getitem_224 = None
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
        r_q_18 = q_18.reshape(16, 14, 14, 64)
        rel_h_18 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_18, Rh_18)
        Rh_18 = None
        rel_w_18 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_18, Rw_18)
        r_q_18 = Rw_18 = None
        getitem_226 = rel_h_18[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_18 = None
        getitem_227 = rel_w_18[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_18 = None
        attn_bias_36 = getitem_226 + getitem_227
        getitem_226 = getitem_227 = None
        attn_bias_37 = attn_bias_36.reshape(-1, 196, 196)
        attn_bias_36 = None
        x_371 = torch._C._nn.scaled_dot_product_attention(
            q_18, k_18, v_18, attn_mask=attn_bias_37, dropout_p=0.0
        )
        q_18 = k_18 = v_18 = attn_bias_37 = None
        view_117 = x_371.view(1, 16, 196, -1)
        x_371 = None
        transpose_18 = view_117.transpose(1, 2)
        view_117 = None
        x_372 = transpose_18.reshape(1, 196, -1)
        transpose_18 = None
        x_373 = torch._C._nn.linear(
            x_372,
            l_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_bias_,
        )
        x_372 = l_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_374 = torch.nn.functional.dropout(x_373, 0.0, False, False)
        x_373 = None
        x_375 = x_374.view(1, 14, 14, -1)
        x_374 = None
        x_376 = x_375.view(1, 1, 1, 14, 14, -1)
        x_375 = None
        permute_65 = x_376.permute(0, 1, 3, 2, 4, 5)
        x_376 = None
        contiguous_46 = permute_65.contiguous()
        permute_65 = None
        x_377 = contiguous_46.view(1, 14, 14, -1)
        contiguous_46 = None
        getitem_228 = x_377[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_377 = None
        x_378 = getitem_228.contiguous()
        getitem_228 = None
        x_379 = x_366 + x_378
        x_366 = x_378 = None
        x_380 = x_379.reshape(1, 196, -1)
        x_379 = None
        layer_norm_37 = torch.nn.functional.layer_norm(
            x_380,
            (1024,),
            l_self_modules_blocks_modules_18_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_18_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_18_modules_norm2_parameters_bias_
        ) = None
        x_381 = torch._C._nn.linear(
            layer_norm_37,
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_37 = (
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_382 = torch._C._nn.gelu(x_381, approximate="none")
        x_381 = None
        x_383 = torch.nn.functional.dropout(x_382, 0.0, False, False)
        x_382 = None
        x_384 = torch._C._nn.linear(
            x_383,
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_383 = (
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_385 = torch.nn.functional.dropout(x_384, 0.0, False, False)
        x_384 = None
        x_386 = x_380 + x_385
        x_380 = x_385 = None
        x_387 = x_386.reshape(1, 14, 14, -1)
        x_386 = None
        x_388 = torch.nn.functional.layer_norm(
            x_387,
            (1024,),
            l_self_modules_blocks_modules_19_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_19_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_19_modules_norm1_parameters_bias_
        ) = None
        x_389 = torch._C._nn.pad(x_388, (0, 0, 0, 0, 0, 0), "constant", None)
        x_388 = None
        x_390 = x_389.view(1, 1, 14, 1, 14, 1024)
        x_389 = None
        permute_66 = x_390.permute(0, 1, 3, 2, 4, 5)
        x_390 = None
        contiguous_48 = permute_66.contiguous()
        permute_66 = None
        windows_16 = contiguous_48.view(-1, 14, 14, 1024)
        contiguous_48 = None
        x_391 = windows_16.reshape(1, 196, -1)
        windows_16 = None
        linear_76 = torch._C._nn.linear(
            x_391,
            l_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_bias_,
        )
        x_391 = (
            l_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_123 = linear_76.view(1, 196, 3, 16, -1)
        linear_76 = None
        qkv_19 = view_123.permute(2, 0, 3, 1, 4)
        view_123 = None
        reshape_147 = qkv_19.reshape(3, 16, 196, -1)
        qkv_19 = None
        unbind_19 = reshape_147.unbind(0)
        reshape_147 = None
        q_19 = unbind_19[0]
        k_19 = unbind_19[1]
        v_19 = unbind_19[2]
        unbind_19 = None
        arange_76 = torch.arange(14)
        getitem_232 = arange_76[(slice(None, None, None), None)]
        arange_76 = None
        q_coords_38 = getitem_232 * 1.0
        getitem_232 = None
        arange_77 = torch.arange(14)
        getitem_233 = arange_77[(None, slice(None, None, None))]
        arange_77 = None
        k_coords_38 = getitem_233 * 1.0
        getitem_233 = None
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
        getitem_235 = arange_78[(slice(None, None, None), None)]
        arange_78 = None
        q_coords_39 = getitem_235 * 1.0
        getitem_235 = None
        arange_79 = torch.arange(14)
        getitem_236 = arange_79[(None, slice(None, None, None))]
        arange_79 = None
        k_coords_39 = getitem_236 * 1.0
        getitem_236 = None
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
        r_q_19 = q_19.reshape(16, 14, 14, 64)
        rel_h_19 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_19, Rh_19)
        Rh_19 = None
        rel_w_19 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_19, Rw_19)
        r_q_19 = Rw_19 = None
        getitem_238 = rel_h_19[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_19 = None
        getitem_239 = rel_w_19[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_19 = None
        attn_bias_38 = getitem_238 + getitem_239
        getitem_238 = getitem_239 = None
        attn_bias_39 = attn_bias_38.reshape(-1, 196, 196)
        attn_bias_38 = None
        x_392 = torch._C._nn.scaled_dot_product_attention(
            q_19, k_19, v_19, attn_mask=attn_bias_39, dropout_p=0.0
        )
        q_19 = k_19 = v_19 = attn_bias_39 = None
        view_124 = x_392.view(1, 16, 196, -1)
        x_392 = None
        transpose_19 = view_124.transpose(1, 2)
        view_124 = None
        x_393 = transpose_19.reshape(1, 196, -1)
        transpose_19 = None
        x_394 = torch._C._nn.linear(
            x_393,
            l_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_bias_,
        )
        x_393 = l_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_395 = torch.nn.functional.dropout(x_394, 0.0, False, False)
        x_394 = None
        x_396 = x_395.view(1, 14, 14, -1)
        x_395 = None
        x_397 = x_396.view(1, 1, 1, 14, 14, -1)
        x_396 = None
        permute_68 = x_397.permute(0, 1, 3, 2, 4, 5)
        x_397 = None
        contiguous_49 = permute_68.contiguous()
        permute_68 = None
        x_398 = contiguous_49.view(1, 14, 14, -1)
        contiguous_49 = None
        getitem_240 = x_398[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_398 = None
        x_399 = getitem_240.contiguous()
        getitem_240 = None
        x_400 = x_387 + x_399
        x_387 = x_399 = None
        x_401 = x_400.reshape(1, 196, -1)
        x_400 = None
        layer_norm_39 = torch.nn.functional.layer_norm(
            x_401,
            (1024,),
            l_self_modules_blocks_modules_19_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_19_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_19_modules_norm2_parameters_bias_
        ) = None
        x_402 = torch._C._nn.linear(
            layer_norm_39,
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_39 = (
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_403 = torch._C._nn.gelu(x_402, approximate="none")
        x_402 = None
        x_404 = torch.nn.functional.dropout(x_403, 0.0, False, False)
        x_403 = None
        x_405 = torch._C._nn.linear(
            x_404,
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_404 = (
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_406 = torch.nn.functional.dropout(x_405, 0.0, False, False)
        x_405 = None
        x_407 = x_401 + x_406
        x_401 = x_406 = None
        x_408 = x_407.reshape(1, 14, 14, -1)
        x_407 = None
        x_409 = torch.nn.functional.layer_norm(
            x_408,
            (1024,),
            l_self_modules_blocks_modules_20_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_20_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_20_modules_norm1_parameters_bias_
        ) = None
        x_410 = torch._C._nn.pad(x_409, (0, 0, 0, 0, 0, 0), "constant", None)
        x_409 = None
        x_411 = x_410.view(1, 1, 14, 1, 14, 1024)
        x_410 = None
        permute_69 = x_411.permute(0, 1, 3, 2, 4, 5)
        x_411 = None
        contiguous_51 = permute_69.contiguous()
        permute_69 = None
        windows_17 = contiguous_51.view(-1, 14, 14, 1024)
        contiguous_51 = None
        x_412 = windows_17.reshape(1, 196, -1)
        windows_17 = None
        linear_80 = torch._C._nn.linear(
            x_412,
            l_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_bias_,
        )
        x_412 = (
            l_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_130 = linear_80.view(1, 196, 3, 16, -1)
        linear_80 = None
        qkv_20 = view_130.permute(2, 0, 3, 1, 4)
        view_130 = None
        reshape_154 = qkv_20.reshape(3, 16, 196, -1)
        qkv_20 = None
        unbind_20 = reshape_154.unbind(0)
        reshape_154 = None
        q_20 = unbind_20[0]
        k_20 = unbind_20[1]
        v_20 = unbind_20[2]
        unbind_20 = None
        arange_80 = torch.arange(14)
        getitem_244 = arange_80[(slice(None, None, None), None)]
        arange_80 = None
        q_coords_40 = getitem_244 * 1.0
        getitem_244 = None
        arange_81 = torch.arange(14)
        getitem_245 = arange_81[(None, slice(None, None, None))]
        arange_81 = None
        k_coords_40 = getitem_245 * 1.0
        getitem_245 = None
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
        getitem_247 = arange_82[(slice(None, None, None), None)]
        arange_82 = None
        q_coords_41 = getitem_247 * 1.0
        getitem_247 = None
        arange_83 = torch.arange(14)
        getitem_248 = arange_83[(None, slice(None, None, None))]
        arange_83 = None
        k_coords_41 = getitem_248 * 1.0
        getitem_248 = None
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
        r_q_20 = q_20.reshape(16, 14, 14, 64)
        rel_h_20 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_20, Rh_20)
        Rh_20 = None
        rel_w_20 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_20, Rw_20)
        r_q_20 = Rw_20 = None
        getitem_250 = rel_h_20[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_20 = None
        getitem_251 = rel_w_20[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_20 = None
        attn_bias_40 = getitem_250 + getitem_251
        getitem_250 = getitem_251 = None
        attn_bias_41 = attn_bias_40.reshape(-1, 196, 196)
        attn_bias_40 = None
        x_413 = torch._C._nn.scaled_dot_product_attention(
            q_20, k_20, v_20, attn_mask=attn_bias_41, dropout_p=0.0
        )
        q_20 = k_20 = v_20 = attn_bias_41 = None
        view_131 = x_413.view(1, 16, 196, -1)
        x_413 = None
        transpose_20 = view_131.transpose(1, 2)
        view_131 = None
        x_414 = transpose_20.reshape(1, 196, -1)
        transpose_20 = None
        x_415 = torch._C._nn.linear(
            x_414,
            l_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_bias_,
        )
        x_414 = l_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_416 = torch.nn.functional.dropout(x_415, 0.0, False, False)
        x_415 = None
        x_417 = x_416.view(1, 14, 14, -1)
        x_416 = None
        x_418 = x_417.view(1, 1, 1, 14, 14, -1)
        x_417 = None
        permute_71 = x_418.permute(0, 1, 3, 2, 4, 5)
        x_418 = None
        contiguous_52 = permute_71.contiguous()
        permute_71 = None
        x_419 = contiguous_52.view(1, 14, 14, -1)
        contiguous_52 = None
        getitem_252 = x_419[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_419 = None
        x_420 = getitem_252.contiguous()
        getitem_252 = None
        x_421 = x_408 + x_420
        x_408 = x_420 = None
        x_422 = x_421.reshape(1, 196, -1)
        x_421 = None
        layer_norm_41 = torch.nn.functional.layer_norm(
            x_422,
            (1024,),
            l_self_modules_blocks_modules_20_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_20_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_20_modules_norm2_parameters_bias_
        ) = None
        x_423 = torch._C._nn.linear(
            layer_norm_41,
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_41 = (
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_424 = torch._C._nn.gelu(x_423, approximate="none")
        x_423 = None
        x_425 = torch.nn.functional.dropout(x_424, 0.0, False, False)
        x_424 = None
        x_426 = torch._C._nn.linear(
            x_425,
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_425 = (
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_427 = torch.nn.functional.dropout(x_426, 0.0, False, False)
        x_426 = None
        x_428 = x_422 + x_427
        x_422 = x_427 = None
        x_429 = x_428.reshape(1, 14, 14, -1)
        x_428 = None
        x_430 = torch.nn.functional.layer_norm(
            x_429,
            (1024,),
            l_self_modules_blocks_modules_21_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_21_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_21_modules_norm1_parameters_bias_
        ) = None
        x_431 = torch._C._nn.pad(x_430, (0, 0, 0, 0, 0, 0), "constant", None)
        x_430 = None
        x_432 = x_431.view(1, 1, 14, 1, 14, 1024)
        x_431 = None
        permute_72 = x_432.permute(0, 1, 3, 2, 4, 5)
        x_432 = None
        contiguous_54 = permute_72.contiguous()
        permute_72 = None
        windows_18 = contiguous_54.view(-1, 14, 14, 1024)
        contiguous_54 = None
        x_433 = windows_18.reshape(1, 196, -1)
        windows_18 = None
        linear_84 = torch._C._nn.linear(
            x_433,
            l_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_bias_,
        )
        x_433 = (
            l_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_137 = linear_84.view(1, 196, 3, 16, -1)
        linear_84 = None
        qkv_21 = view_137.permute(2, 0, 3, 1, 4)
        view_137 = None
        reshape_161 = qkv_21.reshape(3, 16, 196, -1)
        qkv_21 = None
        unbind_21 = reshape_161.unbind(0)
        reshape_161 = None
        q_21 = unbind_21[0]
        k_21 = unbind_21[1]
        v_21 = unbind_21[2]
        unbind_21 = None
        arange_84 = torch.arange(14)
        getitem_256 = arange_84[(slice(None, None, None), None)]
        arange_84 = None
        q_coords_42 = getitem_256 * 1.0
        getitem_256 = None
        arange_85 = torch.arange(14)
        getitem_257 = arange_85[(None, slice(None, None, None))]
        arange_85 = None
        k_coords_42 = getitem_257 * 1.0
        getitem_257 = None
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
        getitem_259 = arange_86[(slice(None, None, None), None)]
        arange_86 = None
        q_coords_43 = getitem_259 * 1.0
        getitem_259 = None
        arange_87 = torch.arange(14)
        getitem_260 = arange_87[(None, slice(None, None, None))]
        arange_87 = None
        k_coords_43 = getitem_260 * 1.0
        getitem_260 = None
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
        r_q_21 = q_21.reshape(16, 14, 14, 64)
        rel_h_21 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_21, Rh_21)
        Rh_21 = None
        rel_w_21 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_21, Rw_21)
        r_q_21 = Rw_21 = None
        getitem_262 = rel_h_21[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_21 = None
        getitem_263 = rel_w_21[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_21 = None
        attn_bias_42 = getitem_262 + getitem_263
        getitem_262 = getitem_263 = None
        attn_bias_43 = attn_bias_42.reshape(-1, 196, 196)
        attn_bias_42 = None
        x_434 = torch._C._nn.scaled_dot_product_attention(
            q_21, k_21, v_21, attn_mask=attn_bias_43, dropout_p=0.0
        )
        q_21 = k_21 = v_21 = attn_bias_43 = None
        view_138 = x_434.view(1, 16, 196, -1)
        x_434 = None
        transpose_21 = view_138.transpose(1, 2)
        view_138 = None
        x_435 = transpose_21.reshape(1, 196, -1)
        transpose_21 = None
        x_436 = torch._C._nn.linear(
            x_435,
            l_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_bias_,
        )
        x_435 = l_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_437 = torch.nn.functional.dropout(x_436, 0.0, False, False)
        x_436 = None
        x_438 = x_437.view(1, 14, 14, -1)
        x_437 = None
        x_439 = x_438.view(1, 1, 1, 14, 14, -1)
        x_438 = None
        permute_74 = x_439.permute(0, 1, 3, 2, 4, 5)
        x_439 = None
        contiguous_55 = permute_74.contiguous()
        permute_74 = None
        x_440 = contiguous_55.view(1, 14, 14, -1)
        contiguous_55 = None
        getitem_264 = x_440[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_440 = None
        x_441 = getitem_264.contiguous()
        getitem_264 = None
        x_442 = x_429 + x_441
        x_429 = x_441 = None
        x_443 = x_442.reshape(1, 196, -1)
        x_442 = None
        layer_norm_43 = torch.nn.functional.layer_norm(
            x_443,
            (1024,),
            l_self_modules_blocks_modules_21_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_21_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_21_modules_norm2_parameters_bias_
        ) = None
        x_444 = torch._C._nn.linear(
            layer_norm_43,
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_43 = (
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_445 = torch._C._nn.gelu(x_444, approximate="none")
        x_444 = None
        x_446 = torch.nn.functional.dropout(x_445, 0.0, False, False)
        x_445 = None
        x_447 = torch._C._nn.linear(
            x_446,
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_446 = (
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_448 = torch.nn.functional.dropout(x_447, 0.0, False, False)
        x_447 = None
        x_449 = x_443 + x_448
        x_443 = x_448 = None
        x_450 = x_449.reshape(1, 14, 14, -1)
        x_449 = None
        x_451 = torch.nn.functional.layer_norm(
            x_450,
            (1024,),
            l_self_modules_blocks_modules_22_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_22_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_22_modules_norm1_parameters_bias_
        ) = None
        x_452 = torch._C._nn.pad(x_451, (0, 0, 0, 0, 0, 0), "constant", None)
        x_451 = None
        x_453 = x_452.view(1, 1, 14, 1, 14, 1024)
        x_452 = None
        permute_75 = x_453.permute(0, 1, 3, 2, 4, 5)
        x_453 = None
        contiguous_57 = permute_75.contiguous()
        permute_75 = None
        windows_19 = contiguous_57.view(-1, 14, 14, 1024)
        contiguous_57 = None
        x_454 = windows_19.reshape(1, 196, -1)
        windows_19 = None
        linear_88 = torch._C._nn.linear(
            x_454,
            l_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_bias_,
        )
        x_454 = (
            l_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_144 = linear_88.view(1, 196, 3, 16, -1)
        linear_88 = None
        qkv_22 = view_144.permute(2, 0, 3, 1, 4)
        view_144 = None
        reshape_168 = qkv_22.reshape(3, 16, 196, -1)
        qkv_22 = None
        unbind_22 = reshape_168.unbind(0)
        reshape_168 = None
        q_22 = unbind_22[0]
        k_22 = unbind_22[1]
        v_22 = unbind_22[2]
        unbind_22 = None
        arange_88 = torch.arange(14)
        getitem_268 = arange_88[(slice(None, None, None), None)]
        arange_88 = None
        q_coords_44 = getitem_268 * 1.0
        getitem_268 = None
        arange_89 = torch.arange(14)
        getitem_269 = arange_89[(None, slice(None, None, None))]
        arange_89 = None
        k_coords_44 = getitem_269 * 1.0
        getitem_269 = None
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
        getitem_271 = arange_90[(slice(None, None, None), None)]
        arange_90 = None
        q_coords_45 = getitem_271 * 1.0
        getitem_271 = None
        arange_91 = torch.arange(14)
        getitem_272 = arange_91[(None, slice(None, None, None))]
        arange_91 = None
        k_coords_45 = getitem_272 * 1.0
        getitem_272 = None
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
        r_q_22 = q_22.reshape(16, 14, 14, 64)
        rel_h_22 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_22, Rh_22)
        Rh_22 = None
        rel_w_22 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_22, Rw_22)
        r_q_22 = Rw_22 = None
        getitem_274 = rel_h_22[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_22 = None
        getitem_275 = rel_w_22[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_22 = None
        attn_bias_44 = getitem_274 + getitem_275
        getitem_274 = getitem_275 = None
        attn_bias_45 = attn_bias_44.reshape(-1, 196, 196)
        attn_bias_44 = None
        x_455 = torch._C._nn.scaled_dot_product_attention(
            q_22, k_22, v_22, attn_mask=attn_bias_45, dropout_p=0.0
        )
        q_22 = k_22 = v_22 = attn_bias_45 = None
        view_145 = x_455.view(1, 16, 196, -1)
        x_455 = None
        transpose_22 = view_145.transpose(1, 2)
        view_145 = None
        x_456 = transpose_22.reshape(1, 196, -1)
        transpose_22 = None
        x_457 = torch._C._nn.linear(
            x_456,
            l_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_bias_,
        )
        x_456 = l_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_458 = torch.nn.functional.dropout(x_457, 0.0, False, False)
        x_457 = None
        x_459 = x_458.view(1, 14, 14, -1)
        x_458 = None
        x_460 = x_459.view(1, 1, 1, 14, 14, -1)
        x_459 = None
        permute_77 = x_460.permute(0, 1, 3, 2, 4, 5)
        x_460 = None
        contiguous_58 = permute_77.contiguous()
        permute_77 = None
        x_461 = contiguous_58.view(1, 14, 14, -1)
        contiguous_58 = None
        getitem_276 = x_461[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_461 = None
        x_462 = getitem_276.contiguous()
        getitem_276 = None
        x_463 = x_450 + x_462
        x_450 = x_462 = None
        x_464 = x_463.reshape(1, 196, -1)
        x_463 = None
        layer_norm_45 = torch.nn.functional.layer_norm(
            x_464,
            (1024,),
            l_self_modules_blocks_modules_22_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_22_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_22_modules_norm2_parameters_bias_
        ) = None
        x_465 = torch._C._nn.linear(
            layer_norm_45,
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_45 = (
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_466 = torch._C._nn.gelu(x_465, approximate="none")
        x_465 = None
        x_467 = torch.nn.functional.dropout(x_466, 0.0, False, False)
        x_466 = None
        x_468 = torch._C._nn.linear(
            x_467,
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_467 = (
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_469 = torch.nn.functional.dropout(x_468, 0.0, False, False)
        x_468 = None
        x_470 = x_464 + x_469
        x_464 = x_469 = None
        x_471 = x_470.reshape(1, 14, 14, -1)
        x_470 = None
        x_472 = torch.nn.functional.layer_norm(
            x_471,
            (1024,),
            l_self_modules_blocks_modules_23_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_23_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_23_modules_norm1_parameters_bias_
        ) = None
        x_473 = x_472.reshape(1, 196, -1)
        x_472 = None
        linear_92 = torch._C._nn.linear(
            x_473,
            l_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_bias_,
        )
        x_473 = (
            l_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_149 = linear_92.view(1, 196, 3, 16, -1)
        linear_92 = None
        qkv_23 = view_149.permute(2, 0, 3, 1, 4)
        view_149 = None
        reshape_175 = qkv_23.reshape(3, 16, 196, -1)
        qkv_23 = None
        unbind_23 = reshape_175.unbind(0)
        reshape_175 = None
        q_23 = unbind_23[0]
        k_23 = unbind_23[1]
        v_23 = unbind_23[2]
        unbind_23 = None
        reshape_176 = (
            l_self_modules_blocks_modules_23_modules_attn_parameters_rel_pos_h_.reshape(
                1, 127, -1
            )
        )
        l_self_modules_blocks_modules_23_modules_attn_parameters_rel_pos_h_ = None
        permute_79 = reshape_176.permute(0, 2, 1)
        reshape_176 = None
        rel_pos_resized_12 = torch.nn.functional.interpolate(
            permute_79, size=27, mode="linear"
        )
        permute_79 = None
        reshape_177 = rel_pos_resized_12.reshape(-1, 27)
        rel_pos_resized_12 = None
        rel_pos_resized_13 = reshape_177.permute(1, 0)
        reshape_177 = None
        arange_92 = torch.arange(14)
        getitem_280 = arange_92[(slice(None, None, None), None)]
        arange_92 = None
        q_coords_46 = getitem_280 * 1.0
        getitem_280 = None
        arange_93 = torch.arange(14)
        getitem_281 = arange_93[(None, slice(None, None, None))]
        arange_93 = None
        k_coords_46 = getitem_281 * 1.0
        getitem_281 = None
        sub_46 = q_coords_46 - k_coords_46
        q_coords_46 = k_coords_46 = None
        relative_coords_46 = sub_46 + 13.0
        sub_46 = None
        long_46 = relative_coords_46.long()
        relative_coords_46 = None
        Rh_23 = rel_pos_resized_13[long_46]
        rel_pos_resized_13 = long_46 = None
        reshape_178 = (
            l_self_modules_blocks_modules_23_modules_attn_parameters_rel_pos_w_.reshape(
                1, 127, -1
            )
        )
        l_self_modules_blocks_modules_23_modules_attn_parameters_rel_pos_w_ = None
        permute_81 = reshape_178.permute(0, 2, 1)
        reshape_178 = None
        rel_pos_resized_14 = torch.nn.functional.interpolate(
            permute_81, size=27, mode="linear"
        )
        permute_81 = None
        reshape_179 = rel_pos_resized_14.reshape(-1, 27)
        rel_pos_resized_14 = None
        rel_pos_resized_15 = reshape_179.permute(1, 0)
        reshape_179 = None
        arange_94 = torch.arange(14)
        getitem_283 = arange_94[(slice(None, None, None), None)]
        arange_94 = None
        q_coords_47 = getitem_283 * 1.0
        getitem_283 = None
        arange_95 = torch.arange(14)
        getitem_284 = arange_95[(None, slice(None, None, None))]
        arange_95 = None
        k_coords_47 = getitem_284 * 1.0
        getitem_284 = None
        sub_47 = q_coords_47 - k_coords_47
        q_coords_47 = k_coords_47 = None
        relative_coords_47 = sub_47 + 13.0
        sub_47 = None
        long_47 = relative_coords_47.long()
        relative_coords_47 = None
        Rw_23 = rel_pos_resized_15[long_47]
        rel_pos_resized_15 = long_47 = None
        r_q_23 = q_23.reshape(16, 14, 14, 64)
        rel_h_23 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_23, Rh_23)
        Rh_23 = None
        rel_w_23 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_23, Rw_23)
        r_q_23 = Rw_23 = None
        getitem_286 = rel_h_23[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_23 = None
        getitem_287 = rel_w_23[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_23 = None
        attn_bias_46 = getitem_286 + getitem_287
        getitem_286 = getitem_287 = None
        attn_bias_47 = attn_bias_46.reshape(-1, 196, 196)
        attn_bias_46 = None
        x_474 = torch._C._nn.scaled_dot_product_attention(
            q_23, k_23, v_23, attn_mask=attn_bias_47, dropout_p=0.0
        )
        q_23 = k_23 = v_23 = attn_bias_47 = None
        view_150 = x_474.view(1, 16, 196, -1)
        x_474 = None
        transpose_23 = view_150.transpose(1, 2)
        view_150 = None
        x_475 = transpose_23.reshape(1, 196, -1)
        transpose_23 = None
        x_476 = torch._C._nn.linear(
            x_475,
            l_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_bias_,
        )
        x_475 = l_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_477 = torch.nn.functional.dropout(x_476, 0.0, False, False)
        x_476 = None
        x_478 = x_477.view(1, 14, 14, -1)
        x_477 = None
        x_479 = x_471 + x_478
        x_471 = x_478 = None
        x_480 = x_479.reshape(1, 196, -1)
        x_479 = None
        layer_norm_47 = torch.nn.functional.layer_norm(
            x_480,
            (1024,),
            l_self_modules_blocks_modules_23_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_23_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_23_modules_norm2_parameters_bias_
        ) = None
        x_481 = torch._C._nn.linear(
            layer_norm_47,
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_47 = (
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_482 = torch._C._nn.gelu(x_481, approximate="none")
        x_481 = None
        x_483 = torch.nn.functional.dropout(x_482, 0.0, False, False)
        x_482 = None
        x_484 = torch._C._nn.linear(
            x_483,
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_483 = (
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_485 = torch.nn.functional.dropout(x_484, 0.0, False, False)
        x_484 = None
        x_486 = x_480 + x_485
        x_480 = x_485 = None
        x_487 = x_486.reshape(1, 14, 14, -1)
        x_486 = None
        permute_83 = x_487.permute(0, 3, 1, 2)
        x_487 = None
        input_1 = torch.conv2d(
            permute_83,
            l_self_modules_neck_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        permute_83 = l_self_modules_neck_modules_0_parameters_weight_ = None
        x_488 = input_1.permute(0, 2, 3, 1)
        input_1 = None
        x_489 = torch.nn.functional.layer_norm(
            x_488,
            (256,),
            l_self_modules_neck_modules_1_parameters_weight_,
            l_self_modules_neck_modules_1_parameters_bias_,
            1e-06,
        )
        x_488 = (
            l_self_modules_neck_modules_1_parameters_weight_
        ) = l_self_modules_neck_modules_1_parameters_bias_ = None
        x_490 = x_489.permute(0, 3, 1, 2)
        x_489 = None
        input_2 = torch.conv2d(
            x_490,
            l_self_modules_neck_modules_2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_490 = l_self_modules_neck_modules_2_parameters_weight_ = None
        x_491 = input_2.permute(0, 2, 3, 1)
        input_2 = None
        x_492 = torch.nn.functional.layer_norm(
            x_491,
            (256,),
            l_self_modules_neck_modules_3_parameters_weight_,
            l_self_modules_neck_modules_3_parameters_bias_,
            1e-06,
        )
        x_491 = (
            l_self_modules_neck_modules_3_parameters_weight_
        ) = l_self_modules_neck_modules_3_parameters_bias_ = None
        x_493 = x_492.permute(0, 3, 1, 2)
        x_492 = None
        x_494 = torch.nn.functional.adaptive_avg_pool2d(x_493, 1)
        x_493 = None
        x_495 = x_494.flatten(1, -1)
        x_494 = None
        x_496 = torch.nn.functional.dropout(x_495, 0.0, False, False)
        x_495 = None
        return (x_496,)
