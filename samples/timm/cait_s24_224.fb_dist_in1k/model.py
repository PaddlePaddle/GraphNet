import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_x_: torch.Tensor,
        L_self_modules_patch_embed_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_patch_embed_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_parameters_pos_embed_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_attn_modules_proj_l_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_attn_modules_proj_l_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_attn_modules_proj_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_attn_modules_proj_w_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_1_modules_attn_modules_proj_l_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_attn_modules_proj_l_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_attn_modules_proj_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_attn_modules_proj_w_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_2_modules_attn_modules_proj_l_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_attn_modules_proj_l_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_attn_modules_proj_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_attn_modules_proj_w_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_3_modules_attn_modules_proj_l_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_attn_modules_proj_l_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_attn_modules_proj_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_attn_modules_proj_w_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_4_modules_attn_modules_proj_l_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_attn_modules_proj_l_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_attn_modules_proj_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_attn_modules_proj_w_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_5_modules_attn_modules_proj_l_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_attn_modules_proj_l_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_attn_modules_proj_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_attn_modules_proj_w_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_6_modules_attn_modules_proj_l_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_attn_modules_proj_l_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_attn_modules_proj_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_attn_modules_proj_w_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_7_modules_attn_modules_proj_l_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_attn_modules_proj_l_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_attn_modules_proj_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_attn_modules_proj_w_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_8_modules_attn_modules_proj_l_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_attn_modules_proj_l_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_attn_modules_proj_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_attn_modules_proj_w_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_9_modules_attn_modules_proj_l_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_attn_modules_proj_l_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_attn_modules_proj_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_attn_modules_proj_w_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_10_modules_attn_modules_proj_l_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_attn_modules_proj_l_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_attn_modules_proj_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_attn_modules_proj_w_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_11_modules_attn_modules_proj_l_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_attn_modules_proj_l_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_attn_modules_proj_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_attn_modules_proj_w_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_12_modules_attn_modules_proj_l_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_attn_modules_proj_l_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_attn_modules_proj_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_attn_modules_proj_w_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_13_modules_attn_modules_proj_l_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_attn_modules_proj_l_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_attn_modules_proj_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_attn_modules_proj_w_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_14_modules_attn_modules_proj_l_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_attn_modules_proj_l_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_attn_modules_proj_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_attn_modules_proj_w_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_15_modules_attn_modules_proj_l_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_attn_modules_proj_l_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_attn_modules_proj_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_attn_modules_proj_w_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_16_modules_attn_modules_proj_l_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_attn_modules_proj_l_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_attn_modules_proj_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_attn_modules_proj_w_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_17_modules_attn_modules_proj_l_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_attn_modules_proj_l_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_attn_modules_proj_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_attn_modules_proj_w_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_18_modules_attn_modules_proj_l_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_attn_modules_proj_l_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_attn_modules_proj_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_attn_modules_proj_w_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_19_modules_attn_modules_proj_l_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_attn_modules_proj_l_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_attn_modules_proj_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_attn_modules_proj_w_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_20_modules_attn_modules_proj_l_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_attn_modules_proj_l_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_attn_modules_proj_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_attn_modules_proj_w_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_21_modules_attn_modules_proj_l_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_attn_modules_proj_l_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_attn_modules_proj_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_attn_modules_proj_w_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_22_modules_attn_modules_proj_l_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_attn_modules_proj_l_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_attn_modules_proj_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_attn_modules_proj_w_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_23_modules_attn_modules_proj_l_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_attn_modules_proj_l_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_attn_modules_proj_w_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_attn_modules_proj_w_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_parameters_cls_token_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_token_only_modules_0_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_token_only_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_token_only_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_token_only_modules_0_modules_attn_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_token_only_modules_0_modules_attn_modules_q_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_token_only_modules_0_modules_attn_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_token_only_modules_0_modules_attn_modules_k_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_token_only_modules_0_modules_attn_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_token_only_modules_0_modules_attn_modules_v_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_token_only_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_token_only_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_token_only_modules_0_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_token_only_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_token_only_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_token_only_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_token_only_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_token_only_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_token_only_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_token_only_modules_1_parameters_gamma_1_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_token_only_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_token_only_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_token_only_modules_1_modules_attn_modules_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_token_only_modules_1_modules_attn_modules_q_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_token_only_modules_1_modules_attn_modules_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_token_only_modules_1_modules_attn_modules_k_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_token_only_modules_1_modules_attn_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_token_only_modules_1_modules_attn_modules_v_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_token_only_modules_1_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_token_only_modules_1_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_token_only_modules_1_parameters_gamma_2_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_token_only_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_token_only_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_token_only_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_token_only_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_token_only_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_token_only_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_blocks_modules_0_modules_attn_modules_proj_l_parameters_weight_ = L_self_modules_blocks_modules_0_modules_attn_modules_proj_l_parameters_weight_
        l_self_modules_blocks_modules_0_modules_attn_modules_proj_l_parameters_bias_ = (
            L_self_modules_blocks_modules_0_modules_attn_modules_proj_l_parameters_bias_
        )
        l_self_modules_blocks_modules_0_modules_attn_modules_proj_w_parameters_weight_ = L_self_modules_blocks_modules_0_modules_attn_modules_proj_w_parameters_weight_
        l_self_modules_blocks_modules_0_modules_attn_modules_proj_w_parameters_bias_ = (
            L_self_modules_blocks_modules_0_modules_attn_modules_proj_w_parameters_bias_
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
        l_self_modules_blocks_modules_1_modules_attn_modules_proj_l_parameters_weight_ = L_self_modules_blocks_modules_1_modules_attn_modules_proj_l_parameters_weight_
        l_self_modules_blocks_modules_1_modules_attn_modules_proj_l_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_attn_modules_proj_l_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_attn_modules_proj_w_parameters_weight_ = L_self_modules_blocks_modules_1_modules_attn_modules_proj_w_parameters_weight_
        l_self_modules_blocks_modules_1_modules_attn_modules_proj_w_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_attn_modules_proj_w_parameters_bias_
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
        l_self_modules_blocks_modules_2_modules_attn_modules_proj_l_parameters_weight_ = L_self_modules_blocks_modules_2_modules_attn_modules_proj_l_parameters_weight_
        l_self_modules_blocks_modules_2_modules_attn_modules_proj_l_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_attn_modules_proj_l_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_attn_modules_proj_w_parameters_weight_ = L_self_modules_blocks_modules_2_modules_attn_modules_proj_w_parameters_weight_
        l_self_modules_blocks_modules_2_modules_attn_modules_proj_w_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_attn_modules_proj_w_parameters_bias_
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
        l_self_modules_blocks_modules_3_modules_attn_modules_proj_l_parameters_weight_ = L_self_modules_blocks_modules_3_modules_attn_modules_proj_l_parameters_weight_
        l_self_modules_blocks_modules_3_modules_attn_modules_proj_l_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_attn_modules_proj_l_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_attn_modules_proj_w_parameters_weight_ = L_self_modules_blocks_modules_3_modules_attn_modules_proj_w_parameters_weight_
        l_self_modules_blocks_modules_3_modules_attn_modules_proj_w_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_attn_modules_proj_w_parameters_bias_
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
        l_self_modules_blocks_modules_4_modules_attn_modules_proj_l_parameters_weight_ = L_self_modules_blocks_modules_4_modules_attn_modules_proj_l_parameters_weight_
        l_self_modules_blocks_modules_4_modules_attn_modules_proj_l_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_attn_modules_proj_l_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_attn_modules_proj_w_parameters_weight_ = L_self_modules_blocks_modules_4_modules_attn_modules_proj_w_parameters_weight_
        l_self_modules_blocks_modules_4_modules_attn_modules_proj_w_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_attn_modules_proj_w_parameters_bias_
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
        l_self_modules_blocks_modules_5_modules_attn_modules_proj_l_parameters_weight_ = L_self_modules_blocks_modules_5_modules_attn_modules_proj_l_parameters_weight_
        l_self_modules_blocks_modules_5_modules_attn_modules_proj_l_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_attn_modules_proj_l_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_attn_modules_proj_w_parameters_weight_ = L_self_modules_blocks_modules_5_modules_attn_modules_proj_w_parameters_weight_
        l_self_modules_blocks_modules_5_modules_attn_modules_proj_w_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_attn_modules_proj_w_parameters_bias_
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
        l_self_modules_blocks_modules_6_modules_attn_modules_proj_l_parameters_weight_ = L_self_modules_blocks_modules_6_modules_attn_modules_proj_l_parameters_weight_
        l_self_modules_blocks_modules_6_modules_attn_modules_proj_l_parameters_bias_ = (
            L_self_modules_blocks_modules_6_modules_attn_modules_proj_l_parameters_bias_
        )
        l_self_modules_blocks_modules_6_modules_attn_modules_proj_w_parameters_weight_ = L_self_modules_blocks_modules_6_modules_attn_modules_proj_w_parameters_weight_
        l_self_modules_blocks_modules_6_modules_attn_modules_proj_w_parameters_bias_ = (
            L_self_modules_blocks_modules_6_modules_attn_modules_proj_w_parameters_bias_
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
        l_self_modules_blocks_modules_7_modules_attn_modules_proj_l_parameters_weight_ = L_self_modules_blocks_modules_7_modules_attn_modules_proj_l_parameters_weight_
        l_self_modules_blocks_modules_7_modules_attn_modules_proj_l_parameters_bias_ = (
            L_self_modules_blocks_modules_7_modules_attn_modules_proj_l_parameters_bias_
        )
        l_self_modules_blocks_modules_7_modules_attn_modules_proj_w_parameters_weight_ = L_self_modules_blocks_modules_7_modules_attn_modules_proj_w_parameters_weight_
        l_self_modules_blocks_modules_7_modules_attn_modules_proj_w_parameters_bias_ = (
            L_self_modules_blocks_modules_7_modules_attn_modules_proj_w_parameters_bias_
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
        l_self_modules_blocks_modules_8_modules_attn_modules_proj_l_parameters_weight_ = L_self_modules_blocks_modules_8_modules_attn_modules_proj_l_parameters_weight_
        l_self_modules_blocks_modules_8_modules_attn_modules_proj_l_parameters_bias_ = (
            L_self_modules_blocks_modules_8_modules_attn_modules_proj_l_parameters_bias_
        )
        l_self_modules_blocks_modules_8_modules_attn_modules_proj_w_parameters_weight_ = L_self_modules_blocks_modules_8_modules_attn_modules_proj_w_parameters_weight_
        l_self_modules_blocks_modules_8_modules_attn_modules_proj_w_parameters_bias_ = (
            L_self_modules_blocks_modules_8_modules_attn_modules_proj_w_parameters_bias_
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
        l_self_modules_blocks_modules_9_modules_attn_modules_proj_l_parameters_weight_ = L_self_modules_blocks_modules_9_modules_attn_modules_proj_l_parameters_weight_
        l_self_modules_blocks_modules_9_modules_attn_modules_proj_l_parameters_bias_ = (
            L_self_modules_blocks_modules_9_modules_attn_modules_proj_l_parameters_bias_
        )
        l_self_modules_blocks_modules_9_modules_attn_modules_proj_w_parameters_weight_ = L_self_modules_blocks_modules_9_modules_attn_modules_proj_w_parameters_weight_
        l_self_modules_blocks_modules_9_modules_attn_modules_proj_w_parameters_bias_ = (
            L_self_modules_blocks_modules_9_modules_attn_modules_proj_w_parameters_bias_
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
        l_self_modules_blocks_modules_10_modules_attn_modules_proj_l_parameters_weight_ = L_self_modules_blocks_modules_10_modules_attn_modules_proj_l_parameters_weight_
        l_self_modules_blocks_modules_10_modules_attn_modules_proj_l_parameters_bias_ = L_self_modules_blocks_modules_10_modules_attn_modules_proj_l_parameters_bias_
        l_self_modules_blocks_modules_10_modules_attn_modules_proj_w_parameters_weight_ = L_self_modules_blocks_modules_10_modules_attn_modules_proj_w_parameters_weight_
        l_self_modules_blocks_modules_10_modules_attn_modules_proj_w_parameters_bias_ = L_self_modules_blocks_modules_10_modules_attn_modules_proj_w_parameters_bias_
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
        l_self_modules_blocks_modules_11_modules_attn_modules_proj_l_parameters_weight_ = L_self_modules_blocks_modules_11_modules_attn_modules_proj_l_parameters_weight_
        l_self_modules_blocks_modules_11_modules_attn_modules_proj_l_parameters_bias_ = L_self_modules_blocks_modules_11_modules_attn_modules_proj_l_parameters_bias_
        l_self_modules_blocks_modules_11_modules_attn_modules_proj_w_parameters_weight_ = L_self_modules_blocks_modules_11_modules_attn_modules_proj_w_parameters_weight_
        l_self_modules_blocks_modules_11_modules_attn_modules_proj_w_parameters_bias_ = L_self_modules_blocks_modules_11_modules_attn_modules_proj_w_parameters_bias_
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
        l_self_modules_blocks_modules_12_modules_attn_modules_proj_l_parameters_weight_ = L_self_modules_blocks_modules_12_modules_attn_modules_proj_l_parameters_weight_
        l_self_modules_blocks_modules_12_modules_attn_modules_proj_l_parameters_bias_ = L_self_modules_blocks_modules_12_modules_attn_modules_proj_l_parameters_bias_
        l_self_modules_blocks_modules_12_modules_attn_modules_proj_w_parameters_weight_ = L_self_modules_blocks_modules_12_modules_attn_modules_proj_w_parameters_weight_
        l_self_modules_blocks_modules_12_modules_attn_modules_proj_w_parameters_bias_ = L_self_modules_blocks_modules_12_modules_attn_modules_proj_w_parameters_bias_
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
        l_self_modules_blocks_modules_13_modules_attn_modules_proj_l_parameters_weight_ = L_self_modules_blocks_modules_13_modules_attn_modules_proj_l_parameters_weight_
        l_self_modules_blocks_modules_13_modules_attn_modules_proj_l_parameters_bias_ = L_self_modules_blocks_modules_13_modules_attn_modules_proj_l_parameters_bias_
        l_self_modules_blocks_modules_13_modules_attn_modules_proj_w_parameters_weight_ = L_self_modules_blocks_modules_13_modules_attn_modules_proj_w_parameters_weight_
        l_self_modules_blocks_modules_13_modules_attn_modules_proj_w_parameters_bias_ = L_self_modules_blocks_modules_13_modules_attn_modules_proj_w_parameters_bias_
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
        l_self_modules_blocks_modules_14_modules_attn_modules_proj_l_parameters_weight_ = L_self_modules_blocks_modules_14_modules_attn_modules_proj_l_parameters_weight_
        l_self_modules_blocks_modules_14_modules_attn_modules_proj_l_parameters_bias_ = L_self_modules_blocks_modules_14_modules_attn_modules_proj_l_parameters_bias_
        l_self_modules_blocks_modules_14_modules_attn_modules_proj_w_parameters_weight_ = L_self_modules_blocks_modules_14_modules_attn_modules_proj_w_parameters_weight_
        l_self_modules_blocks_modules_14_modules_attn_modules_proj_w_parameters_bias_ = L_self_modules_blocks_modules_14_modules_attn_modules_proj_w_parameters_bias_
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
        l_self_modules_blocks_modules_15_modules_attn_modules_proj_l_parameters_weight_ = L_self_modules_blocks_modules_15_modules_attn_modules_proj_l_parameters_weight_
        l_self_modules_blocks_modules_15_modules_attn_modules_proj_l_parameters_bias_ = L_self_modules_blocks_modules_15_modules_attn_modules_proj_l_parameters_bias_
        l_self_modules_blocks_modules_15_modules_attn_modules_proj_w_parameters_weight_ = L_self_modules_blocks_modules_15_modules_attn_modules_proj_w_parameters_weight_
        l_self_modules_blocks_modules_15_modules_attn_modules_proj_w_parameters_bias_ = L_self_modules_blocks_modules_15_modules_attn_modules_proj_w_parameters_bias_
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
        l_self_modules_blocks_modules_16_modules_attn_modules_proj_l_parameters_weight_ = L_self_modules_blocks_modules_16_modules_attn_modules_proj_l_parameters_weight_
        l_self_modules_blocks_modules_16_modules_attn_modules_proj_l_parameters_bias_ = L_self_modules_blocks_modules_16_modules_attn_modules_proj_l_parameters_bias_
        l_self_modules_blocks_modules_16_modules_attn_modules_proj_w_parameters_weight_ = L_self_modules_blocks_modules_16_modules_attn_modules_proj_w_parameters_weight_
        l_self_modules_blocks_modules_16_modules_attn_modules_proj_w_parameters_bias_ = L_self_modules_blocks_modules_16_modules_attn_modules_proj_w_parameters_bias_
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
        l_self_modules_blocks_modules_17_modules_attn_modules_proj_l_parameters_weight_ = L_self_modules_blocks_modules_17_modules_attn_modules_proj_l_parameters_weight_
        l_self_modules_blocks_modules_17_modules_attn_modules_proj_l_parameters_bias_ = L_self_modules_blocks_modules_17_modules_attn_modules_proj_l_parameters_bias_
        l_self_modules_blocks_modules_17_modules_attn_modules_proj_w_parameters_weight_ = L_self_modules_blocks_modules_17_modules_attn_modules_proj_w_parameters_weight_
        l_self_modules_blocks_modules_17_modules_attn_modules_proj_w_parameters_bias_ = L_self_modules_blocks_modules_17_modules_attn_modules_proj_w_parameters_bias_
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
        l_self_modules_blocks_modules_18_modules_attn_modules_proj_l_parameters_weight_ = L_self_modules_blocks_modules_18_modules_attn_modules_proj_l_parameters_weight_
        l_self_modules_blocks_modules_18_modules_attn_modules_proj_l_parameters_bias_ = L_self_modules_blocks_modules_18_modules_attn_modules_proj_l_parameters_bias_
        l_self_modules_blocks_modules_18_modules_attn_modules_proj_w_parameters_weight_ = L_self_modules_blocks_modules_18_modules_attn_modules_proj_w_parameters_weight_
        l_self_modules_blocks_modules_18_modules_attn_modules_proj_w_parameters_bias_ = L_self_modules_blocks_modules_18_modules_attn_modules_proj_w_parameters_bias_
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
        l_self_modules_blocks_modules_19_modules_attn_modules_proj_l_parameters_weight_ = L_self_modules_blocks_modules_19_modules_attn_modules_proj_l_parameters_weight_
        l_self_modules_blocks_modules_19_modules_attn_modules_proj_l_parameters_bias_ = L_self_modules_blocks_modules_19_modules_attn_modules_proj_l_parameters_bias_
        l_self_modules_blocks_modules_19_modules_attn_modules_proj_w_parameters_weight_ = L_self_modules_blocks_modules_19_modules_attn_modules_proj_w_parameters_weight_
        l_self_modules_blocks_modules_19_modules_attn_modules_proj_w_parameters_bias_ = L_self_modules_blocks_modules_19_modules_attn_modules_proj_w_parameters_bias_
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
        l_self_modules_blocks_modules_20_modules_attn_modules_proj_l_parameters_weight_ = L_self_modules_blocks_modules_20_modules_attn_modules_proj_l_parameters_weight_
        l_self_modules_blocks_modules_20_modules_attn_modules_proj_l_parameters_bias_ = L_self_modules_blocks_modules_20_modules_attn_modules_proj_l_parameters_bias_
        l_self_modules_blocks_modules_20_modules_attn_modules_proj_w_parameters_weight_ = L_self_modules_blocks_modules_20_modules_attn_modules_proj_w_parameters_weight_
        l_self_modules_blocks_modules_20_modules_attn_modules_proj_w_parameters_bias_ = L_self_modules_blocks_modules_20_modules_attn_modules_proj_w_parameters_bias_
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
        l_self_modules_blocks_modules_21_modules_attn_modules_proj_l_parameters_weight_ = L_self_modules_blocks_modules_21_modules_attn_modules_proj_l_parameters_weight_
        l_self_modules_blocks_modules_21_modules_attn_modules_proj_l_parameters_bias_ = L_self_modules_blocks_modules_21_modules_attn_modules_proj_l_parameters_bias_
        l_self_modules_blocks_modules_21_modules_attn_modules_proj_w_parameters_weight_ = L_self_modules_blocks_modules_21_modules_attn_modules_proj_w_parameters_weight_
        l_self_modules_blocks_modules_21_modules_attn_modules_proj_w_parameters_bias_ = L_self_modules_blocks_modules_21_modules_attn_modules_proj_w_parameters_bias_
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
        l_self_modules_blocks_modules_22_modules_attn_modules_proj_l_parameters_weight_ = L_self_modules_blocks_modules_22_modules_attn_modules_proj_l_parameters_weight_
        l_self_modules_blocks_modules_22_modules_attn_modules_proj_l_parameters_bias_ = L_self_modules_blocks_modules_22_modules_attn_modules_proj_l_parameters_bias_
        l_self_modules_blocks_modules_22_modules_attn_modules_proj_w_parameters_weight_ = L_self_modules_blocks_modules_22_modules_attn_modules_proj_w_parameters_weight_
        l_self_modules_blocks_modules_22_modules_attn_modules_proj_w_parameters_bias_ = L_self_modules_blocks_modules_22_modules_attn_modules_proj_w_parameters_bias_
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
        l_self_modules_blocks_modules_23_modules_attn_modules_proj_l_parameters_weight_ = L_self_modules_blocks_modules_23_modules_attn_modules_proj_l_parameters_weight_
        l_self_modules_blocks_modules_23_modules_attn_modules_proj_l_parameters_bias_ = L_self_modules_blocks_modules_23_modules_attn_modules_proj_l_parameters_bias_
        l_self_modules_blocks_modules_23_modules_attn_modules_proj_w_parameters_weight_ = L_self_modules_blocks_modules_23_modules_attn_modules_proj_w_parameters_weight_
        l_self_modules_blocks_modules_23_modules_attn_modules_proj_w_parameters_bias_ = L_self_modules_blocks_modules_23_modules_attn_modules_proj_w_parameters_bias_
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
        l_self_parameters_cls_token_ = L_self_parameters_cls_token_
        l_self_modules_blocks_token_only_modules_0_parameters_gamma_1_ = (
            L_self_modules_blocks_token_only_modules_0_parameters_gamma_1_
        )
        l_self_modules_blocks_token_only_modules_0_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_token_only_modules_0_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_token_only_modules_0_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_token_only_modules_0_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_token_only_modules_0_modules_attn_modules_q_parameters_weight_ = L_self_modules_blocks_token_only_modules_0_modules_attn_modules_q_parameters_weight_
        l_self_modules_blocks_token_only_modules_0_modules_attn_modules_q_parameters_bias_ = L_self_modules_blocks_token_only_modules_0_modules_attn_modules_q_parameters_bias_
        l_self_modules_blocks_token_only_modules_0_modules_attn_modules_k_parameters_weight_ = L_self_modules_blocks_token_only_modules_0_modules_attn_modules_k_parameters_weight_
        l_self_modules_blocks_token_only_modules_0_modules_attn_modules_k_parameters_bias_ = L_self_modules_blocks_token_only_modules_0_modules_attn_modules_k_parameters_bias_
        l_self_modules_blocks_token_only_modules_0_modules_attn_modules_v_parameters_weight_ = L_self_modules_blocks_token_only_modules_0_modules_attn_modules_v_parameters_weight_
        l_self_modules_blocks_token_only_modules_0_modules_attn_modules_v_parameters_bias_ = L_self_modules_blocks_token_only_modules_0_modules_attn_modules_v_parameters_bias_
        l_self_modules_blocks_token_only_modules_0_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_token_only_modules_0_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_token_only_modules_0_modules_attn_modules_proj_parameters_bias_ = L_self_modules_blocks_token_only_modules_0_modules_attn_modules_proj_parameters_bias_
        l_self_modules_blocks_token_only_modules_0_parameters_gamma_2_ = (
            L_self_modules_blocks_token_only_modules_0_parameters_gamma_2_
        )
        l_self_modules_blocks_token_only_modules_0_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_token_only_modules_0_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_token_only_modules_0_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_token_only_modules_0_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_token_only_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_blocks_token_only_modules_0_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_blocks_token_only_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_blocks_token_only_modules_0_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_blocks_token_only_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_blocks_token_only_modules_0_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_blocks_token_only_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_blocks_token_only_modules_0_modules_mlp_modules_fc2_parameters_bias_
        l_self_modules_blocks_token_only_modules_1_parameters_gamma_1_ = (
            L_self_modules_blocks_token_only_modules_1_parameters_gamma_1_
        )
        l_self_modules_blocks_token_only_modules_1_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_token_only_modules_1_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_token_only_modules_1_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_token_only_modules_1_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_token_only_modules_1_modules_attn_modules_q_parameters_weight_ = L_self_modules_blocks_token_only_modules_1_modules_attn_modules_q_parameters_weight_
        l_self_modules_blocks_token_only_modules_1_modules_attn_modules_q_parameters_bias_ = L_self_modules_blocks_token_only_modules_1_modules_attn_modules_q_parameters_bias_
        l_self_modules_blocks_token_only_modules_1_modules_attn_modules_k_parameters_weight_ = L_self_modules_blocks_token_only_modules_1_modules_attn_modules_k_parameters_weight_
        l_self_modules_blocks_token_only_modules_1_modules_attn_modules_k_parameters_bias_ = L_self_modules_blocks_token_only_modules_1_modules_attn_modules_k_parameters_bias_
        l_self_modules_blocks_token_only_modules_1_modules_attn_modules_v_parameters_weight_ = L_self_modules_blocks_token_only_modules_1_modules_attn_modules_v_parameters_weight_
        l_self_modules_blocks_token_only_modules_1_modules_attn_modules_v_parameters_bias_ = L_self_modules_blocks_token_only_modules_1_modules_attn_modules_v_parameters_bias_
        l_self_modules_blocks_token_only_modules_1_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_token_only_modules_1_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_token_only_modules_1_modules_attn_modules_proj_parameters_bias_ = L_self_modules_blocks_token_only_modules_1_modules_attn_modules_proj_parameters_bias_
        l_self_modules_blocks_token_only_modules_1_parameters_gamma_2_ = (
            L_self_modules_blocks_token_only_modules_1_parameters_gamma_2_
        )
        l_self_modules_blocks_token_only_modules_1_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_token_only_modules_1_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_token_only_modules_1_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_token_only_modules_1_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_token_only_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_blocks_token_only_modules_1_modules_mlp_modules_fc1_parameters_weight_
        l_self_modules_blocks_token_only_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_blocks_token_only_modules_1_modules_mlp_modules_fc1_parameters_bias_
        l_self_modules_blocks_token_only_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_blocks_token_only_modules_1_modules_mlp_modules_fc2_parameters_weight_
        l_self_modules_blocks_token_only_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_blocks_token_only_modules_1_modules_mlp_modules_fc2_parameters_bias_
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
        x_2 = x_1 + l_self_parameters_pos_embed_
        x_1 = l_self_parameters_pos_embed_ = None
        x_3 = torch.nn.functional.dropout(x_2, 0.0, False, False)
        x_2 = None
        layer_norm = torch.nn.functional.layer_norm(
            x_3,
            (384,),
            l_self_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_0_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_0_modules_norm1_parameters_bias_
        ) = None
        linear = torch._C._nn.linear(
            layer_norm,
            l_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm = (
            l_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape = linear.reshape(1, 196, 3, 8, 48)
        linear = None
        qkv = reshape.permute(2, 0, 3, 1, 4)
        reshape = None
        getitem_4 = qkv[0]
        q = getitem_4 * 0.14433756729740643
        getitem_4 = None
        k = qkv[1]
        v = qkv[2]
        qkv = None
        transpose_1 = k.transpose(-2, -1)
        k = None
        attn = q @ transpose_1
        q = transpose_1 = None
        permute_1 = attn.permute(0, 2, 3, 1)
        attn = None
        linear_1 = torch._C._nn.linear(
            permute_1,
            l_self_modules_blocks_modules_0_modules_attn_modules_proj_l_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_attn_modules_proj_l_parameters_bias_,
        )
        permute_1 = l_self_modules_blocks_modules_0_modules_attn_modules_proj_l_parameters_weight_ = (
            l_self_modules_blocks_modules_0_modules_attn_modules_proj_l_parameters_bias_
        ) = None
        attn_1 = linear_1.permute(0, 3, 1, 2)
        linear_1 = None
        attn_2 = attn_1.softmax(dim=-1)
        attn_1 = None
        permute_3 = attn_2.permute(0, 2, 3, 1)
        attn_2 = None
        linear_2 = torch._C._nn.linear(
            permute_3,
            l_self_modules_blocks_modules_0_modules_attn_modules_proj_w_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_attn_modules_proj_w_parameters_bias_,
        )
        permute_3 = l_self_modules_blocks_modules_0_modules_attn_modules_proj_w_parameters_weight_ = (
            l_self_modules_blocks_modules_0_modules_attn_modules_proj_w_parameters_bias_
        ) = None
        attn_3 = linear_2.permute(0, 3, 1, 2)
        linear_2 = None
        attn_4 = torch.nn.functional.dropout(attn_3, 0.0, False, False)
        attn_3 = None
        matmul_1 = attn_4 @ v
        attn_4 = v = None
        transpose_2 = matmul_1.transpose(1, 2)
        matmul_1 = None
        x_4 = transpose_2.reshape(1, 196, 384)
        transpose_2 = None
        x_5 = torch._C._nn.linear(
            x_4,
            l_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_4 = (
            l_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_6 = torch.nn.functional.dropout(x_5, 0.0, False, False)
        x_5 = None
        mul_1 = l_self_modules_blocks_modules_0_parameters_gamma_1_ * x_6
        l_self_modules_blocks_modules_0_parameters_gamma_1_ = x_6 = None
        x_7 = x_3 + mul_1
        x_3 = mul_1 = None
        layer_norm_1 = torch.nn.functional.layer_norm(
            x_7,
            (384,),
            l_self_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_0_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_0_modules_norm2_parameters_bias_
        ) = None
        x_8 = torch._C._nn.linear(
            layer_norm_1,
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_1 = (
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_9 = torch._C._nn.gelu(x_8, approximate="none")
        x_8 = None
        x_10 = torch.nn.functional.dropout(x_9, 0.0, False, False)
        x_9 = None
        x_11 = torch._C._nn.linear(
            x_10,
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_10 = (
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_12 = torch.nn.functional.dropout(x_11, 0.0, False, False)
        x_11 = None
        mul_2 = l_self_modules_blocks_modules_0_parameters_gamma_2_ * x_12
        l_self_modules_blocks_modules_0_parameters_gamma_2_ = x_12 = None
        x_13 = x_7 + mul_2
        x_7 = mul_2 = None
        layer_norm_2 = torch.nn.functional.layer_norm(
            x_13,
            (384,),
            l_self_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_1_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_1_modules_norm1_parameters_bias_
        ) = None
        linear_6 = torch._C._nn.linear(
            layer_norm_2,
            l_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_2 = (
            l_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_2 = linear_6.reshape(1, 196, 3, 8, 48)
        linear_6 = None
        qkv_1 = reshape_2.permute(2, 0, 3, 1, 4)
        reshape_2 = None
        getitem_7 = qkv_1[0]
        q_1 = getitem_7 * 0.14433756729740643
        getitem_7 = None
        k_1 = qkv_1[1]
        v_1 = qkv_1[2]
        qkv_1 = None
        transpose_3 = k_1.transpose(-2, -1)
        k_1 = None
        attn_5 = q_1 @ transpose_3
        q_1 = transpose_3 = None
        permute_6 = attn_5.permute(0, 2, 3, 1)
        attn_5 = None
        linear_7 = torch._C._nn.linear(
            permute_6,
            l_self_modules_blocks_modules_1_modules_attn_modules_proj_l_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_attn_modules_proj_l_parameters_bias_,
        )
        permute_6 = l_self_modules_blocks_modules_1_modules_attn_modules_proj_l_parameters_weight_ = (
            l_self_modules_blocks_modules_1_modules_attn_modules_proj_l_parameters_bias_
        ) = None
        attn_6 = linear_7.permute(0, 3, 1, 2)
        linear_7 = None
        attn_7 = attn_6.softmax(dim=-1)
        attn_6 = None
        permute_8 = attn_7.permute(0, 2, 3, 1)
        attn_7 = None
        linear_8 = torch._C._nn.linear(
            permute_8,
            l_self_modules_blocks_modules_1_modules_attn_modules_proj_w_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_attn_modules_proj_w_parameters_bias_,
        )
        permute_8 = l_self_modules_blocks_modules_1_modules_attn_modules_proj_w_parameters_weight_ = (
            l_self_modules_blocks_modules_1_modules_attn_modules_proj_w_parameters_bias_
        ) = None
        attn_8 = linear_8.permute(0, 3, 1, 2)
        linear_8 = None
        attn_9 = torch.nn.functional.dropout(attn_8, 0.0, False, False)
        attn_8 = None
        matmul_3 = attn_9 @ v_1
        attn_9 = v_1 = None
        transpose_4 = matmul_3.transpose(1, 2)
        matmul_3 = None
        x_14 = transpose_4.reshape(1, 196, 384)
        transpose_4 = None
        x_15 = torch._C._nn.linear(
            x_14,
            l_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_14 = (
            l_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_16 = torch.nn.functional.dropout(x_15, 0.0, False, False)
        x_15 = None
        mul_4 = l_self_modules_blocks_modules_1_parameters_gamma_1_ * x_16
        l_self_modules_blocks_modules_1_parameters_gamma_1_ = x_16 = None
        x_17 = x_13 + mul_4
        x_13 = mul_4 = None
        layer_norm_3 = torch.nn.functional.layer_norm(
            x_17,
            (384,),
            l_self_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_1_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_1_modules_norm2_parameters_bias_
        ) = None
        x_18 = torch._C._nn.linear(
            layer_norm_3,
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_3 = (
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_19 = torch._C._nn.gelu(x_18, approximate="none")
        x_18 = None
        x_20 = torch.nn.functional.dropout(x_19, 0.0, False, False)
        x_19 = None
        x_21 = torch._C._nn.linear(
            x_20,
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_20 = (
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_22 = torch.nn.functional.dropout(x_21, 0.0, False, False)
        x_21 = None
        mul_5 = l_self_modules_blocks_modules_1_parameters_gamma_2_ * x_22
        l_self_modules_blocks_modules_1_parameters_gamma_2_ = x_22 = None
        x_23 = x_17 + mul_5
        x_17 = mul_5 = None
        layer_norm_4 = torch.nn.functional.layer_norm(
            x_23,
            (384,),
            l_self_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_2_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_2_modules_norm1_parameters_bias_
        ) = None
        linear_12 = torch._C._nn.linear(
            layer_norm_4,
            l_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_4 = (
            l_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_4 = linear_12.reshape(1, 196, 3, 8, 48)
        linear_12 = None
        qkv_2 = reshape_4.permute(2, 0, 3, 1, 4)
        reshape_4 = None
        getitem_10 = qkv_2[0]
        q_2 = getitem_10 * 0.14433756729740643
        getitem_10 = None
        k_2 = qkv_2[1]
        v_2 = qkv_2[2]
        qkv_2 = None
        transpose_5 = k_2.transpose(-2, -1)
        k_2 = None
        attn_10 = q_2 @ transpose_5
        q_2 = transpose_5 = None
        permute_11 = attn_10.permute(0, 2, 3, 1)
        attn_10 = None
        linear_13 = torch._C._nn.linear(
            permute_11,
            l_self_modules_blocks_modules_2_modules_attn_modules_proj_l_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_attn_modules_proj_l_parameters_bias_,
        )
        permute_11 = l_self_modules_blocks_modules_2_modules_attn_modules_proj_l_parameters_weight_ = (
            l_self_modules_blocks_modules_2_modules_attn_modules_proj_l_parameters_bias_
        ) = None
        attn_11 = linear_13.permute(0, 3, 1, 2)
        linear_13 = None
        attn_12 = attn_11.softmax(dim=-1)
        attn_11 = None
        permute_13 = attn_12.permute(0, 2, 3, 1)
        attn_12 = None
        linear_14 = torch._C._nn.linear(
            permute_13,
            l_self_modules_blocks_modules_2_modules_attn_modules_proj_w_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_attn_modules_proj_w_parameters_bias_,
        )
        permute_13 = l_self_modules_blocks_modules_2_modules_attn_modules_proj_w_parameters_weight_ = (
            l_self_modules_blocks_modules_2_modules_attn_modules_proj_w_parameters_bias_
        ) = None
        attn_13 = linear_14.permute(0, 3, 1, 2)
        linear_14 = None
        attn_14 = torch.nn.functional.dropout(attn_13, 0.0, False, False)
        attn_13 = None
        matmul_5 = attn_14 @ v_2
        attn_14 = v_2 = None
        transpose_6 = matmul_5.transpose(1, 2)
        matmul_5 = None
        x_24 = transpose_6.reshape(1, 196, 384)
        transpose_6 = None
        x_25 = torch._C._nn.linear(
            x_24,
            l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_,
        )
        x_24 = (
            l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_26 = torch.nn.functional.dropout(x_25, 0.0, False, False)
        x_25 = None
        mul_7 = l_self_modules_blocks_modules_2_parameters_gamma_1_ * x_26
        l_self_modules_blocks_modules_2_parameters_gamma_1_ = x_26 = None
        x_27 = x_23 + mul_7
        x_23 = mul_7 = None
        layer_norm_5 = torch.nn.functional.layer_norm(
            x_27,
            (384,),
            l_self_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_2_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_2_modules_norm2_parameters_bias_
        ) = None
        x_28 = torch._C._nn.linear(
            layer_norm_5,
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_5 = (
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_29 = torch._C._nn.gelu(x_28, approximate="none")
        x_28 = None
        x_30 = torch.nn.functional.dropout(x_29, 0.0, False, False)
        x_29 = None
        x_31 = torch._C._nn.linear(
            x_30,
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_30 = (
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_32 = torch.nn.functional.dropout(x_31, 0.0, False, False)
        x_31 = None
        mul_8 = l_self_modules_blocks_modules_2_parameters_gamma_2_ * x_32
        l_self_modules_blocks_modules_2_parameters_gamma_2_ = x_32 = None
        x_33 = x_27 + mul_8
        x_27 = mul_8 = None
        layer_norm_6 = torch.nn.functional.layer_norm(
            x_33,
            (384,),
            l_self_modules_blocks_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_3_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_3_modules_norm1_parameters_bias_
        ) = None
        linear_18 = torch._C._nn.linear(
            layer_norm_6,
            l_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_6 = (
            l_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_6 = linear_18.reshape(1, 196, 3, 8, 48)
        linear_18 = None
        qkv_3 = reshape_6.permute(2, 0, 3, 1, 4)
        reshape_6 = None
        getitem_13 = qkv_3[0]
        q_3 = getitem_13 * 0.14433756729740643
        getitem_13 = None
        k_3 = qkv_3[1]
        v_3 = qkv_3[2]
        qkv_3 = None
        transpose_7 = k_3.transpose(-2, -1)
        k_3 = None
        attn_15 = q_3 @ transpose_7
        q_3 = transpose_7 = None
        permute_16 = attn_15.permute(0, 2, 3, 1)
        attn_15 = None
        linear_19 = torch._C._nn.linear(
            permute_16,
            l_self_modules_blocks_modules_3_modules_attn_modules_proj_l_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_attn_modules_proj_l_parameters_bias_,
        )
        permute_16 = l_self_modules_blocks_modules_3_modules_attn_modules_proj_l_parameters_weight_ = (
            l_self_modules_blocks_modules_3_modules_attn_modules_proj_l_parameters_bias_
        ) = None
        attn_16 = linear_19.permute(0, 3, 1, 2)
        linear_19 = None
        attn_17 = attn_16.softmax(dim=-1)
        attn_16 = None
        permute_18 = attn_17.permute(0, 2, 3, 1)
        attn_17 = None
        linear_20 = torch._C._nn.linear(
            permute_18,
            l_self_modules_blocks_modules_3_modules_attn_modules_proj_w_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_attn_modules_proj_w_parameters_bias_,
        )
        permute_18 = l_self_modules_blocks_modules_3_modules_attn_modules_proj_w_parameters_weight_ = (
            l_self_modules_blocks_modules_3_modules_attn_modules_proj_w_parameters_bias_
        ) = None
        attn_18 = linear_20.permute(0, 3, 1, 2)
        linear_20 = None
        attn_19 = torch.nn.functional.dropout(attn_18, 0.0, False, False)
        attn_18 = None
        matmul_7 = attn_19 @ v_3
        attn_19 = v_3 = None
        transpose_8 = matmul_7.transpose(1, 2)
        matmul_7 = None
        x_34 = transpose_8.reshape(1, 196, 384)
        transpose_8 = None
        x_35 = torch._C._nn.linear(
            x_34,
            l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_,
        )
        x_34 = (
            l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_36 = torch.nn.functional.dropout(x_35, 0.0, False, False)
        x_35 = None
        mul_10 = l_self_modules_blocks_modules_3_parameters_gamma_1_ * x_36
        l_self_modules_blocks_modules_3_parameters_gamma_1_ = x_36 = None
        x_37 = x_33 + mul_10
        x_33 = mul_10 = None
        layer_norm_7 = torch.nn.functional.layer_norm(
            x_37,
            (384,),
            l_self_modules_blocks_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_3_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_3_modules_norm2_parameters_bias_
        ) = None
        x_38 = torch._C._nn.linear(
            layer_norm_7,
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_7 = (
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_39 = torch._C._nn.gelu(x_38, approximate="none")
        x_38 = None
        x_40 = torch.nn.functional.dropout(x_39, 0.0, False, False)
        x_39 = None
        x_41 = torch._C._nn.linear(
            x_40,
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_40 = (
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_42 = torch.nn.functional.dropout(x_41, 0.0, False, False)
        x_41 = None
        mul_11 = l_self_modules_blocks_modules_3_parameters_gamma_2_ * x_42
        l_self_modules_blocks_modules_3_parameters_gamma_2_ = x_42 = None
        x_43 = x_37 + mul_11
        x_37 = mul_11 = None
        layer_norm_8 = torch.nn.functional.layer_norm(
            x_43,
            (384,),
            l_self_modules_blocks_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_4_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_4_modules_norm1_parameters_bias_
        ) = None
        linear_24 = torch._C._nn.linear(
            layer_norm_8,
            l_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_8 = (
            l_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_8 = linear_24.reshape(1, 196, 3, 8, 48)
        linear_24 = None
        qkv_4 = reshape_8.permute(2, 0, 3, 1, 4)
        reshape_8 = None
        getitem_16 = qkv_4[0]
        q_4 = getitem_16 * 0.14433756729740643
        getitem_16 = None
        k_4 = qkv_4[1]
        v_4 = qkv_4[2]
        qkv_4 = None
        transpose_9 = k_4.transpose(-2, -1)
        k_4 = None
        attn_20 = q_4 @ transpose_9
        q_4 = transpose_9 = None
        permute_21 = attn_20.permute(0, 2, 3, 1)
        attn_20 = None
        linear_25 = torch._C._nn.linear(
            permute_21,
            l_self_modules_blocks_modules_4_modules_attn_modules_proj_l_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_attn_modules_proj_l_parameters_bias_,
        )
        permute_21 = l_self_modules_blocks_modules_4_modules_attn_modules_proj_l_parameters_weight_ = (
            l_self_modules_blocks_modules_4_modules_attn_modules_proj_l_parameters_bias_
        ) = None
        attn_21 = linear_25.permute(0, 3, 1, 2)
        linear_25 = None
        attn_22 = attn_21.softmax(dim=-1)
        attn_21 = None
        permute_23 = attn_22.permute(0, 2, 3, 1)
        attn_22 = None
        linear_26 = torch._C._nn.linear(
            permute_23,
            l_self_modules_blocks_modules_4_modules_attn_modules_proj_w_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_attn_modules_proj_w_parameters_bias_,
        )
        permute_23 = l_self_modules_blocks_modules_4_modules_attn_modules_proj_w_parameters_weight_ = (
            l_self_modules_blocks_modules_4_modules_attn_modules_proj_w_parameters_bias_
        ) = None
        attn_23 = linear_26.permute(0, 3, 1, 2)
        linear_26 = None
        attn_24 = torch.nn.functional.dropout(attn_23, 0.0, False, False)
        attn_23 = None
        matmul_9 = attn_24 @ v_4
        attn_24 = v_4 = None
        transpose_10 = matmul_9.transpose(1, 2)
        matmul_9 = None
        x_44 = transpose_10.reshape(1, 196, 384)
        transpose_10 = None
        x_45 = torch._C._nn.linear(
            x_44,
            l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_,
        )
        x_44 = (
            l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_46 = torch.nn.functional.dropout(x_45, 0.0, False, False)
        x_45 = None
        mul_13 = l_self_modules_blocks_modules_4_parameters_gamma_1_ * x_46
        l_self_modules_blocks_modules_4_parameters_gamma_1_ = x_46 = None
        x_47 = x_43 + mul_13
        x_43 = mul_13 = None
        layer_norm_9 = torch.nn.functional.layer_norm(
            x_47,
            (384,),
            l_self_modules_blocks_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_4_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_4_modules_norm2_parameters_bias_
        ) = None
        x_48 = torch._C._nn.linear(
            layer_norm_9,
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_9 = (
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_49 = torch._C._nn.gelu(x_48, approximate="none")
        x_48 = None
        x_50 = torch.nn.functional.dropout(x_49, 0.0, False, False)
        x_49 = None
        x_51 = torch._C._nn.linear(
            x_50,
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_50 = (
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_52 = torch.nn.functional.dropout(x_51, 0.0, False, False)
        x_51 = None
        mul_14 = l_self_modules_blocks_modules_4_parameters_gamma_2_ * x_52
        l_self_modules_blocks_modules_4_parameters_gamma_2_ = x_52 = None
        x_53 = x_47 + mul_14
        x_47 = mul_14 = None
        layer_norm_10 = torch.nn.functional.layer_norm(
            x_53,
            (384,),
            l_self_modules_blocks_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_5_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_5_modules_norm1_parameters_bias_
        ) = None
        linear_30 = torch._C._nn.linear(
            layer_norm_10,
            l_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_10 = (
            l_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_10 = linear_30.reshape(1, 196, 3, 8, 48)
        linear_30 = None
        qkv_5 = reshape_10.permute(2, 0, 3, 1, 4)
        reshape_10 = None
        getitem_19 = qkv_5[0]
        q_5 = getitem_19 * 0.14433756729740643
        getitem_19 = None
        k_5 = qkv_5[1]
        v_5 = qkv_5[2]
        qkv_5 = None
        transpose_11 = k_5.transpose(-2, -1)
        k_5 = None
        attn_25 = q_5 @ transpose_11
        q_5 = transpose_11 = None
        permute_26 = attn_25.permute(0, 2, 3, 1)
        attn_25 = None
        linear_31 = torch._C._nn.linear(
            permute_26,
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_l_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_l_parameters_bias_,
        )
        permute_26 = l_self_modules_blocks_modules_5_modules_attn_modules_proj_l_parameters_weight_ = (
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_l_parameters_bias_
        ) = None
        attn_26 = linear_31.permute(0, 3, 1, 2)
        linear_31 = None
        attn_27 = attn_26.softmax(dim=-1)
        attn_26 = None
        permute_28 = attn_27.permute(0, 2, 3, 1)
        attn_27 = None
        linear_32 = torch._C._nn.linear(
            permute_28,
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_w_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_w_parameters_bias_,
        )
        permute_28 = l_self_modules_blocks_modules_5_modules_attn_modules_proj_w_parameters_weight_ = (
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_w_parameters_bias_
        ) = None
        attn_28 = linear_32.permute(0, 3, 1, 2)
        linear_32 = None
        attn_29 = torch.nn.functional.dropout(attn_28, 0.0, False, False)
        attn_28 = None
        matmul_11 = attn_29 @ v_5
        attn_29 = v_5 = None
        transpose_12 = matmul_11.transpose(1, 2)
        matmul_11 = None
        x_54 = transpose_12.reshape(1, 196, 384)
        transpose_12 = None
        x_55 = torch._C._nn.linear(
            x_54,
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_,
        )
        x_54 = (
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_56 = torch.nn.functional.dropout(x_55, 0.0, False, False)
        x_55 = None
        mul_16 = l_self_modules_blocks_modules_5_parameters_gamma_1_ * x_56
        l_self_modules_blocks_modules_5_parameters_gamma_1_ = x_56 = None
        x_57 = x_53 + mul_16
        x_53 = mul_16 = None
        layer_norm_11 = torch.nn.functional.layer_norm(
            x_57,
            (384,),
            l_self_modules_blocks_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_5_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_5_modules_norm2_parameters_bias_
        ) = None
        x_58 = torch._C._nn.linear(
            layer_norm_11,
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_11 = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_59 = torch._C._nn.gelu(x_58, approximate="none")
        x_58 = None
        x_60 = torch.nn.functional.dropout(x_59, 0.0, False, False)
        x_59 = None
        x_61 = torch._C._nn.linear(
            x_60,
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_60 = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_62 = torch.nn.functional.dropout(x_61, 0.0, False, False)
        x_61 = None
        mul_17 = l_self_modules_blocks_modules_5_parameters_gamma_2_ * x_62
        l_self_modules_blocks_modules_5_parameters_gamma_2_ = x_62 = None
        x_63 = x_57 + mul_17
        x_57 = mul_17 = None
        layer_norm_12 = torch.nn.functional.layer_norm(
            x_63,
            (384,),
            l_self_modules_blocks_modules_6_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_6_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_6_modules_norm1_parameters_bias_
        ) = None
        linear_36 = torch._C._nn.linear(
            layer_norm_12,
            l_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_12 = (
            l_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_12 = linear_36.reshape(1, 196, 3, 8, 48)
        linear_36 = None
        qkv_6 = reshape_12.permute(2, 0, 3, 1, 4)
        reshape_12 = None
        getitem_22 = qkv_6[0]
        q_6 = getitem_22 * 0.14433756729740643
        getitem_22 = None
        k_6 = qkv_6[1]
        v_6 = qkv_6[2]
        qkv_6 = None
        transpose_13 = k_6.transpose(-2, -1)
        k_6 = None
        attn_30 = q_6 @ transpose_13
        q_6 = transpose_13 = None
        permute_31 = attn_30.permute(0, 2, 3, 1)
        attn_30 = None
        linear_37 = torch._C._nn.linear(
            permute_31,
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_l_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_l_parameters_bias_,
        )
        permute_31 = l_self_modules_blocks_modules_6_modules_attn_modules_proj_l_parameters_weight_ = (
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_l_parameters_bias_
        ) = None
        attn_31 = linear_37.permute(0, 3, 1, 2)
        linear_37 = None
        attn_32 = attn_31.softmax(dim=-1)
        attn_31 = None
        permute_33 = attn_32.permute(0, 2, 3, 1)
        attn_32 = None
        linear_38 = torch._C._nn.linear(
            permute_33,
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_w_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_w_parameters_bias_,
        )
        permute_33 = l_self_modules_blocks_modules_6_modules_attn_modules_proj_w_parameters_weight_ = (
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_w_parameters_bias_
        ) = None
        attn_33 = linear_38.permute(0, 3, 1, 2)
        linear_38 = None
        attn_34 = torch.nn.functional.dropout(attn_33, 0.0, False, False)
        attn_33 = None
        matmul_13 = attn_34 @ v_6
        attn_34 = v_6 = None
        transpose_14 = matmul_13.transpose(1, 2)
        matmul_13 = None
        x_64 = transpose_14.reshape(1, 196, 384)
        transpose_14 = None
        x_65 = torch._C._nn.linear(
            x_64,
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_,
        )
        x_64 = (
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_66 = torch.nn.functional.dropout(x_65, 0.0, False, False)
        x_65 = None
        mul_19 = l_self_modules_blocks_modules_6_parameters_gamma_1_ * x_66
        l_self_modules_blocks_modules_6_parameters_gamma_1_ = x_66 = None
        x_67 = x_63 + mul_19
        x_63 = mul_19 = None
        layer_norm_13 = torch.nn.functional.layer_norm(
            x_67,
            (384,),
            l_self_modules_blocks_modules_6_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_6_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_6_modules_norm2_parameters_bias_
        ) = None
        x_68 = torch._C._nn.linear(
            layer_norm_13,
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_13 = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_69 = torch._C._nn.gelu(x_68, approximate="none")
        x_68 = None
        x_70 = torch.nn.functional.dropout(x_69, 0.0, False, False)
        x_69 = None
        x_71 = torch._C._nn.linear(
            x_70,
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_70 = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_72 = torch.nn.functional.dropout(x_71, 0.0, False, False)
        x_71 = None
        mul_20 = l_self_modules_blocks_modules_6_parameters_gamma_2_ * x_72
        l_self_modules_blocks_modules_6_parameters_gamma_2_ = x_72 = None
        x_73 = x_67 + mul_20
        x_67 = mul_20 = None
        layer_norm_14 = torch.nn.functional.layer_norm(
            x_73,
            (384,),
            l_self_modules_blocks_modules_7_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_7_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_7_modules_norm1_parameters_bias_
        ) = None
        linear_42 = torch._C._nn.linear(
            layer_norm_14,
            l_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_14 = (
            l_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_14 = linear_42.reshape(1, 196, 3, 8, 48)
        linear_42 = None
        qkv_7 = reshape_14.permute(2, 0, 3, 1, 4)
        reshape_14 = None
        getitem_25 = qkv_7[0]
        q_7 = getitem_25 * 0.14433756729740643
        getitem_25 = None
        k_7 = qkv_7[1]
        v_7 = qkv_7[2]
        qkv_7 = None
        transpose_15 = k_7.transpose(-2, -1)
        k_7 = None
        attn_35 = q_7 @ transpose_15
        q_7 = transpose_15 = None
        permute_36 = attn_35.permute(0, 2, 3, 1)
        attn_35 = None
        linear_43 = torch._C._nn.linear(
            permute_36,
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_l_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_l_parameters_bias_,
        )
        permute_36 = l_self_modules_blocks_modules_7_modules_attn_modules_proj_l_parameters_weight_ = (
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_l_parameters_bias_
        ) = None
        attn_36 = linear_43.permute(0, 3, 1, 2)
        linear_43 = None
        attn_37 = attn_36.softmax(dim=-1)
        attn_36 = None
        permute_38 = attn_37.permute(0, 2, 3, 1)
        attn_37 = None
        linear_44 = torch._C._nn.linear(
            permute_38,
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_w_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_w_parameters_bias_,
        )
        permute_38 = l_self_modules_blocks_modules_7_modules_attn_modules_proj_w_parameters_weight_ = (
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_w_parameters_bias_
        ) = None
        attn_38 = linear_44.permute(0, 3, 1, 2)
        linear_44 = None
        attn_39 = torch.nn.functional.dropout(attn_38, 0.0, False, False)
        attn_38 = None
        matmul_15 = attn_39 @ v_7
        attn_39 = v_7 = None
        transpose_16 = matmul_15.transpose(1, 2)
        matmul_15 = None
        x_74 = transpose_16.reshape(1, 196, 384)
        transpose_16 = None
        x_75 = torch._C._nn.linear(
            x_74,
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_,
        )
        x_74 = (
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_76 = torch.nn.functional.dropout(x_75, 0.0, False, False)
        x_75 = None
        mul_22 = l_self_modules_blocks_modules_7_parameters_gamma_1_ * x_76
        l_self_modules_blocks_modules_7_parameters_gamma_1_ = x_76 = None
        x_77 = x_73 + mul_22
        x_73 = mul_22 = None
        layer_norm_15 = torch.nn.functional.layer_norm(
            x_77,
            (384,),
            l_self_modules_blocks_modules_7_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_7_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_7_modules_norm2_parameters_bias_
        ) = None
        x_78 = torch._C._nn.linear(
            layer_norm_15,
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_15 = (
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_79 = torch._C._nn.gelu(x_78, approximate="none")
        x_78 = None
        x_80 = torch.nn.functional.dropout(x_79, 0.0, False, False)
        x_79 = None
        x_81 = torch._C._nn.linear(
            x_80,
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_80 = (
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_82 = torch.nn.functional.dropout(x_81, 0.0, False, False)
        x_81 = None
        mul_23 = l_self_modules_blocks_modules_7_parameters_gamma_2_ * x_82
        l_self_modules_blocks_modules_7_parameters_gamma_2_ = x_82 = None
        x_83 = x_77 + mul_23
        x_77 = mul_23 = None
        layer_norm_16 = torch.nn.functional.layer_norm(
            x_83,
            (384,),
            l_self_modules_blocks_modules_8_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_8_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_8_modules_norm1_parameters_bias_
        ) = None
        linear_48 = torch._C._nn.linear(
            layer_norm_16,
            l_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_16 = (
            l_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_16 = linear_48.reshape(1, 196, 3, 8, 48)
        linear_48 = None
        qkv_8 = reshape_16.permute(2, 0, 3, 1, 4)
        reshape_16 = None
        getitem_28 = qkv_8[0]
        q_8 = getitem_28 * 0.14433756729740643
        getitem_28 = None
        k_8 = qkv_8[1]
        v_8 = qkv_8[2]
        qkv_8 = None
        transpose_17 = k_8.transpose(-2, -1)
        k_8 = None
        attn_40 = q_8 @ transpose_17
        q_8 = transpose_17 = None
        permute_41 = attn_40.permute(0, 2, 3, 1)
        attn_40 = None
        linear_49 = torch._C._nn.linear(
            permute_41,
            l_self_modules_blocks_modules_8_modules_attn_modules_proj_l_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_attn_modules_proj_l_parameters_bias_,
        )
        permute_41 = l_self_modules_blocks_modules_8_modules_attn_modules_proj_l_parameters_weight_ = (
            l_self_modules_blocks_modules_8_modules_attn_modules_proj_l_parameters_bias_
        ) = None
        attn_41 = linear_49.permute(0, 3, 1, 2)
        linear_49 = None
        attn_42 = attn_41.softmax(dim=-1)
        attn_41 = None
        permute_43 = attn_42.permute(0, 2, 3, 1)
        attn_42 = None
        linear_50 = torch._C._nn.linear(
            permute_43,
            l_self_modules_blocks_modules_8_modules_attn_modules_proj_w_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_attn_modules_proj_w_parameters_bias_,
        )
        permute_43 = l_self_modules_blocks_modules_8_modules_attn_modules_proj_w_parameters_weight_ = (
            l_self_modules_blocks_modules_8_modules_attn_modules_proj_w_parameters_bias_
        ) = None
        attn_43 = linear_50.permute(0, 3, 1, 2)
        linear_50 = None
        attn_44 = torch.nn.functional.dropout(attn_43, 0.0, False, False)
        attn_43 = None
        matmul_17 = attn_44 @ v_8
        attn_44 = v_8 = None
        transpose_18 = matmul_17.transpose(1, 2)
        matmul_17 = None
        x_84 = transpose_18.reshape(1, 196, 384)
        transpose_18 = None
        x_85 = torch._C._nn.linear(
            x_84,
            l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_,
        )
        x_84 = (
            l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_86 = torch.nn.functional.dropout(x_85, 0.0, False, False)
        x_85 = None
        mul_25 = l_self_modules_blocks_modules_8_parameters_gamma_1_ * x_86
        l_self_modules_blocks_modules_8_parameters_gamma_1_ = x_86 = None
        x_87 = x_83 + mul_25
        x_83 = mul_25 = None
        layer_norm_17 = torch.nn.functional.layer_norm(
            x_87,
            (384,),
            l_self_modules_blocks_modules_8_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_8_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_8_modules_norm2_parameters_bias_
        ) = None
        x_88 = torch._C._nn.linear(
            layer_norm_17,
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_17 = (
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_89 = torch._C._nn.gelu(x_88, approximate="none")
        x_88 = None
        x_90 = torch.nn.functional.dropout(x_89, 0.0, False, False)
        x_89 = None
        x_91 = torch._C._nn.linear(
            x_90,
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_90 = (
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_92 = torch.nn.functional.dropout(x_91, 0.0, False, False)
        x_91 = None
        mul_26 = l_self_modules_blocks_modules_8_parameters_gamma_2_ * x_92
        l_self_modules_blocks_modules_8_parameters_gamma_2_ = x_92 = None
        x_93 = x_87 + mul_26
        x_87 = mul_26 = None
        layer_norm_18 = torch.nn.functional.layer_norm(
            x_93,
            (384,),
            l_self_modules_blocks_modules_9_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_9_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_9_modules_norm1_parameters_bias_
        ) = None
        linear_54 = torch._C._nn.linear(
            layer_norm_18,
            l_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_18 = (
            l_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_18 = linear_54.reshape(1, 196, 3, 8, 48)
        linear_54 = None
        qkv_9 = reshape_18.permute(2, 0, 3, 1, 4)
        reshape_18 = None
        getitem_31 = qkv_9[0]
        q_9 = getitem_31 * 0.14433756729740643
        getitem_31 = None
        k_9 = qkv_9[1]
        v_9 = qkv_9[2]
        qkv_9 = None
        transpose_19 = k_9.transpose(-2, -1)
        k_9 = None
        attn_45 = q_9 @ transpose_19
        q_9 = transpose_19 = None
        permute_46 = attn_45.permute(0, 2, 3, 1)
        attn_45 = None
        linear_55 = torch._C._nn.linear(
            permute_46,
            l_self_modules_blocks_modules_9_modules_attn_modules_proj_l_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_attn_modules_proj_l_parameters_bias_,
        )
        permute_46 = l_self_modules_blocks_modules_9_modules_attn_modules_proj_l_parameters_weight_ = (
            l_self_modules_blocks_modules_9_modules_attn_modules_proj_l_parameters_bias_
        ) = None
        attn_46 = linear_55.permute(0, 3, 1, 2)
        linear_55 = None
        attn_47 = attn_46.softmax(dim=-1)
        attn_46 = None
        permute_48 = attn_47.permute(0, 2, 3, 1)
        attn_47 = None
        linear_56 = torch._C._nn.linear(
            permute_48,
            l_self_modules_blocks_modules_9_modules_attn_modules_proj_w_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_attn_modules_proj_w_parameters_bias_,
        )
        permute_48 = l_self_modules_blocks_modules_9_modules_attn_modules_proj_w_parameters_weight_ = (
            l_self_modules_blocks_modules_9_modules_attn_modules_proj_w_parameters_bias_
        ) = None
        attn_48 = linear_56.permute(0, 3, 1, 2)
        linear_56 = None
        attn_49 = torch.nn.functional.dropout(attn_48, 0.0, False, False)
        attn_48 = None
        matmul_19 = attn_49 @ v_9
        attn_49 = v_9 = None
        transpose_20 = matmul_19.transpose(1, 2)
        matmul_19 = None
        x_94 = transpose_20.reshape(1, 196, 384)
        transpose_20 = None
        x_95 = torch._C._nn.linear(
            x_94,
            l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_,
        )
        x_94 = (
            l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_96 = torch.nn.functional.dropout(x_95, 0.0, False, False)
        x_95 = None
        mul_28 = l_self_modules_blocks_modules_9_parameters_gamma_1_ * x_96
        l_self_modules_blocks_modules_9_parameters_gamma_1_ = x_96 = None
        x_97 = x_93 + mul_28
        x_93 = mul_28 = None
        layer_norm_19 = torch.nn.functional.layer_norm(
            x_97,
            (384,),
            l_self_modules_blocks_modules_9_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_9_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_9_modules_norm2_parameters_bias_
        ) = None
        x_98 = torch._C._nn.linear(
            layer_norm_19,
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_19 = (
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_99 = torch._C._nn.gelu(x_98, approximate="none")
        x_98 = None
        x_100 = torch.nn.functional.dropout(x_99, 0.0, False, False)
        x_99 = None
        x_101 = torch._C._nn.linear(
            x_100,
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_100 = (
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_102 = torch.nn.functional.dropout(x_101, 0.0, False, False)
        x_101 = None
        mul_29 = l_self_modules_blocks_modules_9_parameters_gamma_2_ * x_102
        l_self_modules_blocks_modules_9_parameters_gamma_2_ = x_102 = None
        x_103 = x_97 + mul_29
        x_97 = mul_29 = None
        layer_norm_20 = torch.nn.functional.layer_norm(
            x_103,
            (384,),
            l_self_modules_blocks_modules_10_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_10_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_10_modules_norm1_parameters_bias_
        ) = None
        linear_60 = torch._C._nn.linear(
            layer_norm_20,
            l_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_20 = (
            l_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_20 = linear_60.reshape(1, 196, 3, 8, 48)
        linear_60 = None
        qkv_10 = reshape_20.permute(2, 0, 3, 1, 4)
        reshape_20 = None
        getitem_34 = qkv_10[0]
        q_10 = getitem_34 * 0.14433756729740643
        getitem_34 = None
        k_10 = qkv_10[1]
        v_10 = qkv_10[2]
        qkv_10 = None
        transpose_21 = k_10.transpose(-2, -1)
        k_10 = None
        attn_50 = q_10 @ transpose_21
        q_10 = transpose_21 = None
        permute_51 = attn_50.permute(0, 2, 3, 1)
        attn_50 = None
        linear_61 = torch._C._nn.linear(
            permute_51,
            l_self_modules_blocks_modules_10_modules_attn_modules_proj_l_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_attn_modules_proj_l_parameters_bias_,
        )
        permute_51 = l_self_modules_blocks_modules_10_modules_attn_modules_proj_l_parameters_weight_ = l_self_modules_blocks_modules_10_modules_attn_modules_proj_l_parameters_bias_ = (None)
        attn_51 = linear_61.permute(0, 3, 1, 2)
        linear_61 = None
        attn_52 = attn_51.softmax(dim=-1)
        attn_51 = None
        permute_53 = attn_52.permute(0, 2, 3, 1)
        attn_52 = None
        linear_62 = torch._C._nn.linear(
            permute_53,
            l_self_modules_blocks_modules_10_modules_attn_modules_proj_w_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_attn_modules_proj_w_parameters_bias_,
        )
        permute_53 = l_self_modules_blocks_modules_10_modules_attn_modules_proj_w_parameters_weight_ = l_self_modules_blocks_modules_10_modules_attn_modules_proj_w_parameters_bias_ = (None)
        attn_53 = linear_62.permute(0, 3, 1, 2)
        linear_62 = None
        attn_54 = torch.nn.functional.dropout(attn_53, 0.0, False, False)
        attn_53 = None
        matmul_21 = attn_54 @ v_10
        attn_54 = v_10 = None
        transpose_22 = matmul_21.transpose(1, 2)
        matmul_21 = None
        x_104 = transpose_22.reshape(1, 196, 384)
        transpose_22 = None
        x_105 = torch._C._nn.linear(
            x_104,
            l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_,
        )
        x_104 = l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_106 = torch.nn.functional.dropout(x_105, 0.0, False, False)
        x_105 = None
        mul_31 = l_self_modules_blocks_modules_10_parameters_gamma_1_ * x_106
        l_self_modules_blocks_modules_10_parameters_gamma_1_ = x_106 = None
        x_107 = x_103 + mul_31
        x_103 = mul_31 = None
        layer_norm_21 = torch.nn.functional.layer_norm(
            x_107,
            (384,),
            l_self_modules_blocks_modules_10_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_10_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_10_modules_norm2_parameters_bias_
        ) = None
        x_108 = torch._C._nn.linear(
            layer_norm_21,
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_21 = (
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_109 = torch._C._nn.gelu(x_108, approximate="none")
        x_108 = None
        x_110 = torch.nn.functional.dropout(x_109, 0.0, False, False)
        x_109 = None
        x_111 = torch._C._nn.linear(
            x_110,
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_110 = (
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_112 = torch.nn.functional.dropout(x_111, 0.0, False, False)
        x_111 = None
        mul_32 = l_self_modules_blocks_modules_10_parameters_gamma_2_ * x_112
        l_self_modules_blocks_modules_10_parameters_gamma_2_ = x_112 = None
        x_113 = x_107 + mul_32
        x_107 = mul_32 = None
        layer_norm_22 = torch.nn.functional.layer_norm(
            x_113,
            (384,),
            l_self_modules_blocks_modules_11_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_11_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_11_modules_norm1_parameters_bias_
        ) = None
        linear_66 = torch._C._nn.linear(
            layer_norm_22,
            l_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_22 = (
            l_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_22 = linear_66.reshape(1, 196, 3, 8, 48)
        linear_66 = None
        qkv_11 = reshape_22.permute(2, 0, 3, 1, 4)
        reshape_22 = None
        getitem_37 = qkv_11[0]
        q_11 = getitem_37 * 0.14433756729740643
        getitem_37 = None
        k_11 = qkv_11[1]
        v_11 = qkv_11[2]
        qkv_11 = None
        transpose_23 = k_11.transpose(-2, -1)
        k_11 = None
        attn_55 = q_11 @ transpose_23
        q_11 = transpose_23 = None
        permute_56 = attn_55.permute(0, 2, 3, 1)
        attn_55 = None
        linear_67 = torch._C._nn.linear(
            permute_56,
            l_self_modules_blocks_modules_11_modules_attn_modules_proj_l_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_attn_modules_proj_l_parameters_bias_,
        )
        permute_56 = l_self_modules_blocks_modules_11_modules_attn_modules_proj_l_parameters_weight_ = l_self_modules_blocks_modules_11_modules_attn_modules_proj_l_parameters_bias_ = (None)
        attn_56 = linear_67.permute(0, 3, 1, 2)
        linear_67 = None
        attn_57 = attn_56.softmax(dim=-1)
        attn_56 = None
        permute_58 = attn_57.permute(0, 2, 3, 1)
        attn_57 = None
        linear_68 = torch._C._nn.linear(
            permute_58,
            l_self_modules_blocks_modules_11_modules_attn_modules_proj_w_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_attn_modules_proj_w_parameters_bias_,
        )
        permute_58 = l_self_modules_blocks_modules_11_modules_attn_modules_proj_w_parameters_weight_ = l_self_modules_blocks_modules_11_modules_attn_modules_proj_w_parameters_bias_ = (None)
        attn_58 = linear_68.permute(0, 3, 1, 2)
        linear_68 = None
        attn_59 = torch.nn.functional.dropout(attn_58, 0.0, False, False)
        attn_58 = None
        matmul_23 = attn_59 @ v_11
        attn_59 = v_11 = None
        transpose_24 = matmul_23.transpose(1, 2)
        matmul_23 = None
        x_114 = transpose_24.reshape(1, 196, 384)
        transpose_24 = None
        x_115 = torch._C._nn.linear(
            x_114,
            l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_,
        )
        x_114 = l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_116 = torch.nn.functional.dropout(x_115, 0.0, False, False)
        x_115 = None
        mul_34 = l_self_modules_blocks_modules_11_parameters_gamma_1_ * x_116
        l_self_modules_blocks_modules_11_parameters_gamma_1_ = x_116 = None
        x_117 = x_113 + mul_34
        x_113 = mul_34 = None
        layer_norm_23 = torch.nn.functional.layer_norm(
            x_117,
            (384,),
            l_self_modules_blocks_modules_11_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_11_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_11_modules_norm2_parameters_bias_
        ) = None
        x_118 = torch._C._nn.linear(
            layer_norm_23,
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_23 = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_119 = torch._C._nn.gelu(x_118, approximate="none")
        x_118 = None
        x_120 = torch.nn.functional.dropout(x_119, 0.0, False, False)
        x_119 = None
        x_121 = torch._C._nn.linear(
            x_120,
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_120 = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_122 = torch.nn.functional.dropout(x_121, 0.0, False, False)
        x_121 = None
        mul_35 = l_self_modules_blocks_modules_11_parameters_gamma_2_ * x_122
        l_self_modules_blocks_modules_11_parameters_gamma_2_ = x_122 = None
        x_123 = x_117 + mul_35
        x_117 = mul_35 = None
        layer_norm_24 = torch.nn.functional.layer_norm(
            x_123,
            (384,),
            l_self_modules_blocks_modules_12_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_12_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_12_modules_norm1_parameters_bias_
        ) = None
        linear_72 = torch._C._nn.linear(
            layer_norm_24,
            l_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_24 = (
            l_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_24 = linear_72.reshape(1, 196, 3, 8, 48)
        linear_72 = None
        qkv_12 = reshape_24.permute(2, 0, 3, 1, 4)
        reshape_24 = None
        getitem_40 = qkv_12[0]
        q_12 = getitem_40 * 0.14433756729740643
        getitem_40 = None
        k_12 = qkv_12[1]
        v_12 = qkv_12[2]
        qkv_12 = None
        transpose_25 = k_12.transpose(-2, -1)
        k_12 = None
        attn_60 = q_12 @ transpose_25
        q_12 = transpose_25 = None
        permute_61 = attn_60.permute(0, 2, 3, 1)
        attn_60 = None
        linear_73 = torch._C._nn.linear(
            permute_61,
            l_self_modules_blocks_modules_12_modules_attn_modules_proj_l_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_attn_modules_proj_l_parameters_bias_,
        )
        permute_61 = l_self_modules_blocks_modules_12_modules_attn_modules_proj_l_parameters_weight_ = l_self_modules_blocks_modules_12_modules_attn_modules_proj_l_parameters_bias_ = (None)
        attn_61 = linear_73.permute(0, 3, 1, 2)
        linear_73 = None
        attn_62 = attn_61.softmax(dim=-1)
        attn_61 = None
        permute_63 = attn_62.permute(0, 2, 3, 1)
        attn_62 = None
        linear_74 = torch._C._nn.linear(
            permute_63,
            l_self_modules_blocks_modules_12_modules_attn_modules_proj_w_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_attn_modules_proj_w_parameters_bias_,
        )
        permute_63 = l_self_modules_blocks_modules_12_modules_attn_modules_proj_w_parameters_weight_ = l_self_modules_blocks_modules_12_modules_attn_modules_proj_w_parameters_bias_ = (None)
        attn_63 = linear_74.permute(0, 3, 1, 2)
        linear_74 = None
        attn_64 = torch.nn.functional.dropout(attn_63, 0.0, False, False)
        attn_63 = None
        matmul_25 = attn_64 @ v_12
        attn_64 = v_12 = None
        transpose_26 = matmul_25.transpose(1, 2)
        matmul_25 = None
        x_124 = transpose_26.reshape(1, 196, 384)
        transpose_26 = None
        x_125 = torch._C._nn.linear(
            x_124,
            l_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_,
        )
        x_124 = l_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_126 = torch.nn.functional.dropout(x_125, 0.0, False, False)
        x_125 = None
        mul_37 = l_self_modules_blocks_modules_12_parameters_gamma_1_ * x_126
        l_self_modules_blocks_modules_12_parameters_gamma_1_ = x_126 = None
        x_127 = x_123 + mul_37
        x_123 = mul_37 = None
        layer_norm_25 = torch.nn.functional.layer_norm(
            x_127,
            (384,),
            l_self_modules_blocks_modules_12_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_12_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_12_modules_norm2_parameters_bias_
        ) = None
        x_128 = torch._C._nn.linear(
            layer_norm_25,
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_25 = (
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_129 = torch._C._nn.gelu(x_128, approximate="none")
        x_128 = None
        x_130 = torch.nn.functional.dropout(x_129, 0.0, False, False)
        x_129 = None
        x_131 = torch._C._nn.linear(
            x_130,
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_130 = (
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_132 = torch.nn.functional.dropout(x_131, 0.0, False, False)
        x_131 = None
        mul_38 = l_self_modules_blocks_modules_12_parameters_gamma_2_ * x_132
        l_self_modules_blocks_modules_12_parameters_gamma_2_ = x_132 = None
        x_133 = x_127 + mul_38
        x_127 = mul_38 = None
        layer_norm_26 = torch.nn.functional.layer_norm(
            x_133,
            (384,),
            l_self_modules_blocks_modules_13_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_13_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_13_modules_norm1_parameters_bias_
        ) = None
        linear_78 = torch._C._nn.linear(
            layer_norm_26,
            l_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_26 = (
            l_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_26 = linear_78.reshape(1, 196, 3, 8, 48)
        linear_78 = None
        qkv_13 = reshape_26.permute(2, 0, 3, 1, 4)
        reshape_26 = None
        getitem_43 = qkv_13[0]
        q_13 = getitem_43 * 0.14433756729740643
        getitem_43 = None
        k_13 = qkv_13[1]
        v_13 = qkv_13[2]
        qkv_13 = None
        transpose_27 = k_13.transpose(-2, -1)
        k_13 = None
        attn_65 = q_13 @ transpose_27
        q_13 = transpose_27 = None
        permute_66 = attn_65.permute(0, 2, 3, 1)
        attn_65 = None
        linear_79 = torch._C._nn.linear(
            permute_66,
            l_self_modules_blocks_modules_13_modules_attn_modules_proj_l_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_attn_modules_proj_l_parameters_bias_,
        )
        permute_66 = l_self_modules_blocks_modules_13_modules_attn_modules_proj_l_parameters_weight_ = l_self_modules_blocks_modules_13_modules_attn_modules_proj_l_parameters_bias_ = (None)
        attn_66 = linear_79.permute(0, 3, 1, 2)
        linear_79 = None
        attn_67 = attn_66.softmax(dim=-1)
        attn_66 = None
        permute_68 = attn_67.permute(0, 2, 3, 1)
        attn_67 = None
        linear_80 = torch._C._nn.linear(
            permute_68,
            l_self_modules_blocks_modules_13_modules_attn_modules_proj_w_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_attn_modules_proj_w_parameters_bias_,
        )
        permute_68 = l_self_modules_blocks_modules_13_modules_attn_modules_proj_w_parameters_weight_ = l_self_modules_blocks_modules_13_modules_attn_modules_proj_w_parameters_bias_ = (None)
        attn_68 = linear_80.permute(0, 3, 1, 2)
        linear_80 = None
        attn_69 = torch.nn.functional.dropout(attn_68, 0.0, False, False)
        attn_68 = None
        matmul_27 = attn_69 @ v_13
        attn_69 = v_13 = None
        transpose_28 = matmul_27.transpose(1, 2)
        matmul_27 = None
        x_134 = transpose_28.reshape(1, 196, 384)
        transpose_28 = None
        x_135 = torch._C._nn.linear(
            x_134,
            l_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_,
        )
        x_134 = l_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_136 = torch.nn.functional.dropout(x_135, 0.0, False, False)
        x_135 = None
        mul_40 = l_self_modules_blocks_modules_13_parameters_gamma_1_ * x_136
        l_self_modules_blocks_modules_13_parameters_gamma_1_ = x_136 = None
        x_137 = x_133 + mul_40
        x_133 = mul_40 = None
        layer_norm_27 = torch.nn.functional.layer_norm(
            x_137,
            (384,),
            l_self_modules_blocks_modules_13_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_13_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_13_modules_norm2_parameters_bias_
        ) = None
        x_138 = torch._C._nn.linear(
            layer_norm_27,
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_27 = (
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_139 = torch._C._nn.gelu(x_138, approximate="none")
        x_138 = None
        x_140 = torch.nn.functional.dropout(x_139, 0.0, False, False)
        x_139 = None
        x_141 = torch._C._nn.linear(
            x_140,
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_140 = (
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_142 = torch.nn.functional.dropout(x_141, 0.0, False, False)
        x_141 = None
        mul_41 = l_self_modules_blocks_modules_13_parameters_gamma_2_ * x_142
        l_self_modules_blocks_modules_13_parameters_gamma_2_ = x_142 = None
        x_143 = x_137 + mul_41
        x_137 = mul_41 = None
        layer_norm_28 = torch.nn.functional.layer_norm(
            x_143,
            (384,),
            l_self_modules_blocks_modules_14_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_14_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_14_modules_norm1_parameters_bias_
        ) = None
        linear_84 = torch._C._nn.linear(
            layer_norm_28,
            l_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_28 = (
            l_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_28 = linear_84.reshape(1, 196, 3, 8, 48)
        linear_84 = None
        qkv_14 = reshape_28.permute(2, 0, 3, 1, 4)
        reshape_28 = None
        getitem_46 = qkv_14[0]
        q_14 = getitem_46 * 0.14433756729740643
        getitem_46 = None
        k_14 = qkv_14[1]
        v_14 = qkv_14[2]
        qkv_14 = None
        transpose_29 = k_14.transpose(-2, -1)
        k_14 = None
        attn_70 = q_14 @ transpose_29
        q_14 = transpose_29 = None
        permute_71 = attn_70.permute(0, 2, 3, 1)
        attn_70 = None
        linear_85 = torch._C._nn.linear(
            permute_71,
            l_self_modules_blocks_modules_14_modules_attn_modules_proj_l_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_attn_modules_proj_l_parameters_bias_,
        )
        permute_71 = l_self_modules_blocks_modules_14_modules_attn_modules_proj_l_parameters_weight_ = l_self_modules_blocks_modules_14_modules_attn_modules_proj_l_parameters_bias_ = (None)
        attn_71 = linear_85.permute(0, 3, 1, 2)
        linear_85 = None
        attn_72 = attn_71.softmax(dim=-1)
        attn_71 = None
        permute_73 = attn_72.permute(0, 2, 3, 1)
        attn_72 = None
        linear_86 = torch._C._nn.linear(
            permute_73,
            l_self_modules_blocks_modules_14_modules_attn_modules_proj_w_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_attn_modules_proj_w_parameters_bias_,
        )
        permute_73 = l_self_modules_blocks_modules_14_modules_attn_modules_proj_w_parameters_weight_ = l_self_modules_blocks_modules_14_modules_attn_modules_proj_w_parameters_bias_ = (None)
        attn_73 = linear_86.permute(0, 3, 1, 2)
        linear_86 = None
        attn_74 = torch.nn.functional.dropout(attn_73, 0.0, False, False)
        attn_73 = None
        matmul_29 = attn_74 @ v_14
        attn_74 = v_14 = None
        transpose_30 = matmul_29.transpose(1, 2)
        matmul_29 = None
        x_144 = transpose_30.reshape(1, 196, 384)
        transpose_30 = None
        x_145 = torch._C._nn.linear(
            x_144,
            l_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_bias_,
        )
        x_144 = l_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_146 = torch.nn.functional.dropout(x_145, 0.0, False, False)
        x_145 = None
        mul_43 = l_self_modules_blocks_modules_14_parameters_gamma_1_ * x_146
        l_self_modules_blocks_modules_14_parameters_gamma_1_ = x_146 = None
        x_147 = x_143 + mul_43
        x_143 = mul_43 = None
        layer_norm_29 = torch.nn.functional.layer_norm(
            x_147,
            (384,),
            l_self_modules_blocks_modules_14_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_14_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_14_modules_norm2_parameters_bias_
        ) = None
        x_148 = torch._C._nn.linear(
            layer_norm_29,
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_29 = (
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_149 = torch._C._nn.gelu(x_148, approximate="none")
        x_148 = None
        x_150 = torch.nn.functional.dropout(x_149, 0.0, False, False)
        x_149 = None
        x_151 = torch._C._nn.linear(
            x_150,
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_150 = (
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_152 = torch.nn.functional.dropout(x_151, 0.0, False, False)
        x_151 = None
        mul_44 = l_self_modules_blocks_modules_14_parameters_gamma_2_ * x_152
        l_self_modules_blocks_modules_14_parameters_gamma_2_ = x_152 = None
        x_153 = x_147 + mul_44
        x_147 = mul_44 = None
        layer_norm_30 = torch.nn.functional.layer_norm(
            x_153,
            (384,),
            l_self_modules_blocks_modules_15_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_15_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_15_modules_norm1_parameters_bias_
        ) = None
        linear_90 = torch._C._nn.linear(
            layer_norm_30,
            l_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_30 = (
            l_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_30 = linear_90.reshape(1, 196, 3, 8, 48)
        linear_90 = None
        qkv_15 = reshape_30.permute(2, 0, 3, 1, 4)
        reshape_30 = None
        getitem_49 = qkv_15[0]
        q_15 = getitem_49 * 0.14433756729740643
        getitem_49 = None
        k_15 = qkv_15[1]
        v_15 = qkv_15[2]
        qkv_15 = None
        transpose_31 = k_15.transpose(-2, -1)
        k_15 = None
        attn_75 = q_15 @ transpose_31
        q_15 = transpose_31 = None
        permute_76 = attn_75.permute(0, 2, 3, 1)
        attn_75 = None
        linear_91 = torch._C._nn.linear(
            permute_76,
            l_self_modules_blocks_modules_15_modules_attn_modules_proj_l_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_attn_modules_proj_l_parameters_bias_,
        )
        permute_76 = l_self_modules_blocks_modules_15_modules_attn_modules_proj_l_parameters_weight_ = l_self_modules_blocks_modules_15_modules_attn_modules_proj_l_parameters_bias_ = (None)
        attn_76 = linear_91.permute(0, 3, 1, 2)
        linear_91 = None
        attn_77 = attn_76.softmax(dim=-1)
        attn_76 = None
        permute_78 = attn_77.permute(0, 2, 3, 1)
        attn_77 = None
        linear_92 = torch._C._nn.linear(
            permute_78,
            l_self_modules_blocks_modules_15_modules_attn_modules_proj_w_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_attn_modules_proj_w_parameters_bias_,
        )
        permute_78 = l_self_modules_blocks_modules_15_modules_attn_modules_proj_w_parameters_weight_ = l_self_modules_blocks_modules_15_modules_attn_modules_proj_w_parameters_bias_ = (None)
        attn_78 = linear_92.permute(0, 3, 1, 2)
        linear_92 = None
        attn_79 = torch.nn.functional.dropout(attn_78, 0.0, False, False)
        attn_78 = None
        matmul_31 = attn_79 @ v_15
        attn_79 = v_15 = None
        transpose_32 = matmul_31.transpose(1, 2)
        matmul_31 = None
        x_154 = transpose_32.reshape(1, 196, 384)
        transpose_32 = None
        x_155 = torch._C._nn.linear(
            x_154,
            l_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_bias_,
        )
        x_154 = l_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_156 = torch.nn.functional.dropout(x_155, 0.0, False, False)
        x_155 = None
        mul_46 = l_self_modules_blocks_modules_15_parameters_gamma_1_ * x_156
        l_self_modules_blocks_modules_15_parameters_gamma_1_ = x_156 = None
        x_157 = x_153 + mul_46
        x_153 = mul_46 = None
        layer_norm_31 = torch.nn.functional.layer_norm(
            x_157,
            (384,),
            l_self_modules_blocks_modules_15_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_15_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_15_modules_norm2_parameters_bias_
        ) = None
        x_158 = torch._C._nn.linear(
            layer_norm_31,
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_31 = (
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_159 = torch._C._nn.gelu(x_158, approximate="none")
        x_158 = None
        x_160 = torch.nn.functional.dropout(x_159, 0.0, False, False)
        x_159 = None
        x_161 = torch._C._nn.linear(
            x_160,
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_160 = (
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_162 = torch.nn.functional.dropout(x_161, 0.0, False, False)
        x_161 = None
        mul_47 = l_self_modules_blocks_modules_15_parameters_gamma_2_ * x_162
        l_self_modules_blocks_modules_15_parameters_gamma_2_ = x_162 = None
        x_163 = x_157 + mul_47
        x_157 = mul_47 = None
        layer_norm_32 = torch.nn.functional.layer_norm(
            x_163,
            (384,),
            l_self_modules_blocks_modules_16_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_16_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_16_modules_norm1_parameters_bias_
        ) = None
        linear_96 = torch._C._nn.linear(
            layer_norm_32,
            l_self_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_32 = (
            l_self_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_32 = linear_96.reshape(1, 196, 3, 8, 48)
        linear_96 = None
        qkv_16 = reshape_32.permute(2, 0, 3, 1, 4)
        reshape_32 = None
        getitem_52 = qkv_16[0]
        q_16 = getitem_52 * 0.14433756729740643
        getitem_52 = None
        k_16 = qkv_16[1]
        v_16 = qkv_16[2]
        qkv_16 = None
        transpose_33 = k_16.transpose(-2, -1)
        k_16 = None
        attn_80 = q_16 @ transpose_33
        q_16 = transpose_33 = None
        permute_81 = attn_80.permute(0, 2, 3, 1)
        attn_80 = None
        linear_97 = torch._C._nn.linear(
            permute_81,
            l_self_modules_blocks_modules_16_modules_attn_modules_proj_l_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_attn_modules_proj_l_parameters_bias_,
        )
        permute_81 = l_self_modules_blocks_modules_16_modules_attn_modules_proj_l_parameters_weight_ = l_self_modules_blocks_modules_16_modules_attn_modules_proj_l_parameters_bias_ = (None)
        attn_81 = linear_97.permute(0, 3, 1, 2)
        linear_97 = None
        attn_82 = attn_81.softmax(dim=-1)
        attn_81 = None
        permute_83 = attn_82.permute(0, 2, 3, 1)
        attn_82 = None
        linear_98 = torch._C._nn.linear(
            permute_83,
            l_self_modules_blocks_modules_16_modules_attn_modules_proj_w_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_attn_modules_proj_w_parameters_bias_,
        )
        permute_83 = l_self_modules_blocks_modules_16_modules_attn_modules_proj_w_parameters_weight_ = l_self_modules_blocks_modules_16_modules_attn_modules_proj_w_parameters_bias_ = (None)
        attn_83 = linear_98.permute(0, 3, 1, 2)
        linear_98 = None
        attn_84 = torch.nn.functional.dropout(attn_83, 0.0, False, False)
        attn_83 = None
        matmul_33 = attn_84 @ v_16
        attn_84 = v_16 = None
        transpose_34 = matmul_33.transpose(1, 2)
        matmul_33 = None
        x_164 = transpose_34.reshape(1, 196, 384)
        transpose_34 = None
        x_165 = torch._C._nn.linear(
            x_164,
            l_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_bias_,
        )
        x_164 = l_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_166 = torch.nn.functional.dropout(x_165, 0.0, False, False)
        x_165 = None
        mul_49 = l_self_modules_blocks_modules_16_parameters_gamma_1_ * x_166
        l_self_modules_blocks_modules_16_parameters_gamma_1_ = x_166 = None
        x_167 = x_163 + mul_49
        x_163 = mul_49 = None
        layer_norm_33 = torch.nn.functional.layer_norm(
            x_167,
            (384,),
            l_self_modules_blocks_modules_16_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_16_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_16_modules_norm2_parameters_bias_
        ) = None
        x_168 = torch._C._nn.linear(
            layer_norm_33,
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_33 = (
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_169 = torch._C._nn.gelu(x_168, approximate="none")
        x_168 = None
        x_170 = torch.nn.functional.dropout(x_169, 0.0, False, False)
        x_169 = None
        x_171 = torch._C._nn.linear(
            x_170,
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_170 = (
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_172 = torch.nn.functional.dropout(x_171, 0.0, False, False)
        x_171 = None
        mul_50 = l_self_modules_blocks_modules_16_parameters_gamma_2_ * x_172
        l_self_modules_blocks_modules_16_parameters_gamma_2_ = x_172 = None
        x_173 = x_167 + mul_50
        x_167 = mul_50 = None
        layer_norm_34 = torch.nn.functional.layer_norm(
            x_173,
            (384,),
            l_self_modules_blocks_modules_17_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_17_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_17_modules_norm1_parameters_bias_
        ) = None
        linear_102 = torch._C._nn.linear(
            layer_norm_34,
            l_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_34 = (
            l_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_34 = linear_102.reshape(1, 196, 3, 8, 48)
        linear_102 = None
        qkv_17 = reshape_34.permute(2, 0, 3, 1, 4)
        reshape_34 = None
        getitem_55 = qkv_17[0]
        q_17 = getitem_55 * 0.14433756729740643
        getitem_55 = None
        k_17 = qkv_17[1]
        v_17 = qkv_17[2]
        qkv_17 = None
        transpose_35 = k_17.transpose(-2, -1)
        k_17 = None
        attn_85 = q_17 @ transpose_35
        q_17 = transpose_35 = None
        permute_86 = attn_85.permute(0, 2, 3, 1)
        attn_85 = None
        linear_103 = torch._C._nn.linear(
            permute_86,
            l_self_modules_blocks_modules_17_modules_attn_modules_proj_l_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_attn_modules_proj_l_parameters_bias_,
        )
        permute_86 = l_self_modules_blocks_modules_17_modules_attn_modules_proj_l_parameters_weight_ = l_self_modules_blocks_modules_17_modules_attn_modules_proj_l_parameters_bias_ = (None)
        attn_86 = linear_103.permute(0, 3, 1, 2)
        linear_103 = None
        attn_87 = attn_86.softmax(dim=-1)
        attn_86 = None
        permute_88 = attn_87.permute(0, 2, 3, 1)
        attn_87 = None
        linear_104 = torch._C._nn.linear(
            permute_88,
            l_self_modules_blocks_modules_17_modules_attn_modules_proj_w_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_attn_modules_proj_w_parameters_bias_,
        )
        permute_88 = l_self_modules_blocks_modules_17_modules_attn_modules_proj_w_parameters_weight_ = l_self_modules_blocks_modules_17_modules_attn_modules_proj_w_parameters_bias_ = (None)
        attn_88 = linear_104.permute(0, 3, 1, 2)
        linear_104 = None
        attn_89 = torch.nn.functional.dropout(attn_88, 0.0, False, False)
        attn_88 = None
        matmul_35 = attn_89 @ v_17
        attn_89 = v_17 = None
        transpose_36 = matmul_35.transpose(1, 2)
        matmul_35 = None
        x_174 = transpose_36.reshape(1, 196, 384)
        transpose_36 = None
        x_175 = torch._C._nn.linear(
            x_174,
            l_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_bias_,
        )
        x_174 = l_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_176 = torch.nn.functional.dropout(x_175, 0.0, False, False)
        x_175 = None
        mul_52 = l_self_modules_blocks_modules_17_parameters_gamma_1_ * x_176
        l_self_modules_blocks_modules_17_parameters_gamma_1_ = x_176 = None
        x_177 = x_173 + mul_52
        x_173 = mul_52 = None
        layer_norm_35 = torch.nn.functional.layer_norm(
            x_177,
            (384,),
            l_self_modules_blocks_modules_17_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_17_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_17_modules_norm2_parameters_bias_
        ) = None
        x_178 = torch._C._nn.linear(
            layer_norm_35,
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_35 = (
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_179 = torch._C._nn.gelu(x_178, approximate="none")
        x_178 = None
        x_180 = torch.nn.functional.dropout(x_179, 0.0, False, False)
        x_179 = None
        x_181 = torch._C._nn.linear(
            x_180,
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_180 = (
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_182 = torch.nn.functional.dropout(x_181, 0.0, False, False)
        x_181 = None
        mul_53 = l_self_modules_blocks_modules_17_parameters_gamma_2_ * x_182
        l_self_modules_blocks_modules_17_parameters_gamma_2_ = x_182 = None
        x_183 = x_177 + mul_53
        x_177 = mul_53 = None
        layer_norm_36 = torch.nn.functional.layer_norm(
            x_183,
            (384,),
            l_self_modules_blocks_modules_18_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_18_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_18_modules_norm1_parameters_bias_
        ) = None
        linear_108 = torch._C._nn.linear(
            layer_norm_36,
            l_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_36 = (
            l_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_36 = linear_108.reshape(1, 196, 3, 8, 48)
        linear_108 = None
        qkv_18 = reshape_36.permute(2, 0, 3, 1, 4)
        reshape_36 = None
        getitem_58 = qkv_18[0]
        q_18 = getitem_58 * 0.14433756729740643
        getitem_58 = None
        k_18 = qkv_18[1]
        v_18 = qkv_18[2]
        qkv_18 = None
        transpose_37 = k_18.transpose(-2, -1)
        k_18 = None
        attn_90 = q_18 @ transpose_37
        q_18 = transpose_37 = None
        permute_91 = attn_90.permute(0, 2, 3, 1)
        attn_90 = None
        linear_109 = torch._C._nn.linear(
            permute_91,
            l_self_modules_blocks_modules_18_modules_attn_modules_proj_l_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_attn_modules_proj_l_parameters_bias_,
        )
        permute_91 = l_self_modules_blocks_modules_18_modules_attn_modules_proj_l_parameters_weight_ = l_self_modules_blocks_modules_18_modules_attn_modules_proj_l_parameters_bias_ = (None)
        attn_91 = linear_109.permute(0, 3, 1, 2)
        linear_109 = None
        attn_92 = attn_91.softmax(dim=-1)
        attn_91 = None
        permute_93 = attn_92.permute(0, 2, 3, 1)
        attn_92 = None
        linear_110 = torch._C._nn.linear(
            permute_93,
            l_self_modules_blocks_modules_18_modules_attn_modules_proj_w_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_attn_modules_proj_w_parameters_bias_,
        )
        permute_93 = l_self_modules_blocks_modules_18_modules_attn_modules_proj_w_parameters_weight_ = l_self_modules_blocks_modules_18_modules_attn_modules_proj_w_parameters_bias_ = (None)
        attn_93 = linear_110.permute(0, 3, 1, 2)
        linear_110 = None
        attn_94 = torch.nn.functional.dropout(attn_93, 0.0, False, False)
        attn_93 = None
        matmul_37 = attn_94 @ v_18
        attn_94 = v_18 = None
        transpose_38 = matmul_37.transpose(1, 2)
        matmul_37 = None
        x_184 = transpose_38.reshape(1, 196, 384)
        transpose_38 = None
        x_185 = torch._C._nn.linear(
            x_184,
            l_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_bias_,
        )
        x_184 = l_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_186 = torch.nn.functional.dropout(x_185, 0.0, False, False)
        x_185 = None
        mul_55 = l_self_modules_blocks_modules_18_parameters_gamma_1_ * x_186
        l_self_modules_blocks_modules_18_parameters_gamma_1_ = x_186 = None
        x_187 = x_183 + mul_55
        x_183 = mul_55 = None
        layer_norm_37 = torch.nn.functional.layer_norm(
            x_187,
            (384,),
            l_self_modules_blocks_modules_18_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_18_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_18_modules_norm2_parameters_bias_
        ) = None
        x_188 = torch._C._nn.linear(
            layer_norm_37,
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_37 = (
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_189 = torch._C._nn.gelu(x_188, approximate="none")
        x_188 = None
        x_190 = torch.nn.functional.dropout(x_189, 0.0, False, False)
        x_189 = None
        x_191 = torch._C._nn.linear(
            x_190,
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_190 = (
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_192 = torch.nn.functional.dropout(x_191, 0.0, False, False)
        x_191 = None
        mul_56 = l_self_modules_blocks_modules_18_parameters_gamma_2_ * x_192
        l_self_modules_blocks_modules_18_parameters_gamma_2_ = x_192 = None
        x_193 = x_187 + mul_56
        x_187 = mul_56 = None
        layer_norm_38 = torch.nn.functional.layer_norm(
            x_193,
            (384,),
            l_self_modules_blocks_modules_19_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_19_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_19_modules_norm1_parameters_bias_
        ) = None
        linear_114 = torch._C._nn.linear(
            layer_norm_38,
            l_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_38 = (
            l_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_38 = linear_114.reshape(1, 196, 3, 8, 48)
        linear_114 = None
        qkv_19 = reshape_38.permute(2, 0, 3, 1, 4)
        reshape_38 = None
        getitem_61 = qkv_19[0]
        q_19 = getitem_61 * 0.14433756729740643
        getitem_61 = None
        k_19 = qkv_19[1]
        v_19 = qkv_19[2]
        qkv_19 = None
        transpose_39 = k_19.transpose(-2, -1)
        k_19 = None
        attn_95 = q_19 @ transpose_39
        q_19 = transpose_39 = None
        permute_96 = attn_95.permute(0, 2, 3, 1)
        attn_95 = None
        linear_115 = torch._C._nn.linear(
            permute_96,
            l_self_modules_blocks_modules_19_modules_attn_modules_proj_l_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_attn_modules_proj_l_parameters_bias_,
        )
        permute_96 = l_self_modules_blocks_modules_19_modules_attn_modules_proj_l_parameters_weight_ = l_self_modules_blocks_modules_19_modules_attn_modules_proj_l_parameters_bias_ = (None)
        attn_96 = linear_115.permute(0, 3, 1, 2)
        linear_115 = None
        attn_97 = attn_96.softmax(dim=-1)
        attn_96 = None
        permute_98 = attn_97.permute(0, 2, 3, 1)
        attn_97 = None
        linear_116 = torch._C._nn.linear(
            permute_98,
            l_self_modules_blocks_modules_19_modules_attn_modules_proj_w_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_attn_modules_proj_w_parameters_bias_,
        )
        permute_98 = l_self_modules_blocks_modules_19_modules_attn_modules_proj_w_parameters_weight_ = l_self_modules_blocks_modules_19_modules_attn_modules_proj_w_parameters_bias_ = (None)
        attn_98 = linear_116.permute(0, 3, 1, 2)
        linear_116 = None
        attn_99 = torch.nn.functional.dropout(attn_98, 0.0, False, False)
        attn_98 = None
        matmul_39 = attn_99 @ v_19
        attn_99 = v_19 = None
        transpose_40 = matmul_39.transpose(1, 2)
        matmul_39 = None
        x_194 = transpose_40.reshape(1, 196, 384)
        transpose_40 = None
        x_195 = torch._C._nn.linear(
            x_194,
            l_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_bias_,
        )
        x_194 = l_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_196 = torch.nn.functional.dropout(x_195, 0.0, False, False)
        x_195 = None
        mul_58 = l_self_modules_blocks_modules_19_parameters_gamma_1_ * x_196
        l_self_modules_blocks_modules_19_parameters_gamma_1_ = x_196 = None
        x_197 = x_193 + mul_58
        x_193 = mul_58 = None
        layer_norm_39 = torch.nn.functional.layer_norm(
            x_197,
            (384,),
            l_self_modules_blocks_modules_19_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_19_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_19_modules_norm2_parameters_bias_
        ) = None
        x_198 = torch._C._nn.linear(
            layer_norm_39,
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_39 = (
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_199 = torch._C._nn.gelu(x_198, approximate="none")
        x_198 = None
        x_200 = torch.nn.functional.dropout(x_199, 0.0, False, False)
        x_199 = None
        x_201 = torch._C._nn.linear(
            x_200,
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_200 = (
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_202 = torch.nn.functional.dropout(x_201, 0.0, False, False)
        x_201 = None
        mul_59 = l_self_modules_blocks_modules_19_parameters_gamma_2_ * x_202
        l_self_modules_blocks_modules_19_parameters_gamma_2_ = x_202 = None
        x_203 = x_197 + mul_59
        x_197 = mul_59 = None
        layer_norm_40 = torch.nn.functional.layer_norm(
            x_203,
            (384,),
            l_self_modules_blocks_modules_20_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_20_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_20_modules_norm1_parameters_bias_
        ) = None
        linear_120 = torch._C._nn.linear(
            layer_norm_40,
            l_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_40 = (
            l_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_40 = linear_120.reshape(1, 196, 3, 8, 48)
        linear_120 = None
        qkv_20 = reshape_40.permute(2, 0, 3, 1, 4)
        reshape_40 = None
        getitem_64 = qkv_20[0]
        q_20 = getitem_64 * 0.14433756729740643
        getitem_64 = None
        k_20 = qkv_20[1]
        v_20 = qkv_20[2]
        qkv_20 = None
        transpose_41 = k_20.transpose(-2, -1)
        k_20 = None
        attn_100 = q_20 @ transpose_41
        q_20 = transpose_41 = None
        permute_101 = attn_100.permute(0, 2, 3, 1)
        attn_100 = None
        linear_121 = torch._C._nn.linear(
            permute_101,
            l_self_modules_blocks_modules_20_modules_attn_modules_proj_l_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_attn_modules_proj_l_parameters_bias_,
        )
        permute_101 = l_self_modules_blocks_modules_20_modules_attn_modules_proj_l_parameters_weight_ = l_self_modules_blocks_modules_20_modules_attn_modules_proj_l_parameters_bias_ = (None)
        attn_101 = linear_121.permute(0, 3, 1, 2)
        linear_121 = None
        attn_102 = attn_101.softmax(dim=-1)
        attn_101 = None
        permute_103 = attn_102.permute(0, 2, 3, 1)
        attn_102 = None
        linear_122 = torch._C._nn.linear(
            permute_103,
            l_self_modules_blocks_modules_20_modules_attn_modules_proj_w_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_attn_modules_proj_w_parameters_bias_,
        )
        permute_103 = l_self_modules_blocks_modules_20_modules_attn_modules_proj_w_parameters_weight_ = l_self_modules_blocks_modules_20_modules_attn_modules_proj_w_parameters_bias_ = (None)
        attn_103 = linear_122.permute(0, 3, 1, 2)
        linear_122 = None
        attn_104 = torch.nn.functional.dropout(attn_103, 0.0, False, False)
        attn_103 = None
        matmul_41 = attn_104 @ v_20
        attn_104 = v_20 = None
        transpose_42 = matmul_41.transpose(1, 2)
        matmul_41 = None
        x_204 = transpose_42.reshape(1, 196, 384)
        transpose_42 = None
        x_205 = torch._C._nn.linear(
            x_204,
            l_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_bias_,
        )
        x_204 = l_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_206 = torch.nn.functional.dropout(x_205, 0.0, False, False)
        x_205 = None
        mul_61 = l_self_modules_blocks_modules_20_parameters_gamma_1_ * x_206
        l_self_modules_blocks_modules_20_parameters_gamma_1_ = x_206 = None
        x_207 = x_203 + mul_61
        x_203 = mul_61 = None
        layer_norm_41 = torch.nn.functional.layer_norm(
            x_207,
            (384,),
            l_self_modules_blocks_modules_20_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_20_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_20_modules_norm2_parameters_bias_
        ) = None
        x_208 = torch._C._nn.linear(
            layer_norm_41,
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_41 = (
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_209 = torch._C._nn.gelu(x_208, approximate="none")
        x_208 = None
        x_210 = torch.nn.functional.dropout(x_209, 0.0, False, False)
        x_209 = None
        x_211 = torch._C._nn.linear(
            x_210,
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_210 = (
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_212 = torch.nn.functional.dropout(x_211, 0.0, False, False)
        x_211 = None
        mul_62 = l_self_modules_blocks_modules_20_parameters_gamma_2_ * x_212
        l_self_modules_blocks_modules_20_parameters_gamma_2_ = x_212 = None
        x_213 = x_207 + mul_62
        x_207 = mul_62 = None
        layer_norm_42 = torch.nn.functional.layer_norm(
            x_213,
            (384,),
            l_self_modules_blocks_modules_21_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_21_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_21_modules_norm1_parameters_bias_
        ) = None
        linear_126 = torch._C._nn.linear(
            layer_norm_42,
            l_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_42 = (
            l_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_42 = linear_126.reshape(1, 196, 3, 8, 48)
        linear_126 = None
        qkv_21 = reshape_42.permute(2, 0, 3, 1, 4)
        reshape_42 = None
        getitem_67 = qkv_21[0]
        q_21 = getitem_67 * 0.14433756729740643
        getitem_67 = None
        k_21 = qkv_21[1]
        v_21 = qkv_21[2]
        qkv_21 = None
        transpose_43 = k_21.transpose(-2, -1)
        k_21 = None
        attn_105 = q_21 @ transpose_43
        q_21 = transpose_43 = None
        permute_106 = attn_105.permute(0, 2, 3, 1)
        attn_105 = None
        linear_127 = torch._C._nn.linear(
            permute_106,
            l_self_modules_blocks_modules_21_modules_attn_modules_proj_l_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_attn_modules_proj_l_parameters_bias_,
        )
        permute_106 = l_self_modules_blocks_modules_21_modules_attn_modules_proj_l_parameters_weight_ = l_self_modules_blocks_modules_21_modules_attn_modules_proj_l_parameters_bias_ = (None)
        attn_106 = linear_127.permute(0, 3, 1, 2)
        linear_127 = None
        attn_107 = attn_106.softmax(dim=-1)
        attn_106 = None
        permute_108 = attn_107.permute(0, 2, 3, 1)
        attn_107 = None
        linear_128 = torch._C._nn.linear(
            permute_108,
            l_self_modules_blocks_modules_21_modules_attn_modules_proj_w_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_attn_modules_proj_w_parameters_bias_,
        )
        permute_108 = l_self_modules_blocks_modules_21_modules_attn_modules_proj_w_parameters_weight_ = l_self_modules_blocks_modules_21_modules_attn_modules_proj_w_parameters_bias_ = (None)
        attn_108 = linear_128.permute(0, 3, 1, 2)
        linear_128 = None
        attn_109 = torch.nn.functional.dropout(attn_108, 0.0, False, False)
        attn_108 = None
        matmul_43 = attn_109 @ v_21
        attn_109 = v_21 = None
        transpose_44 = matmul_43.transpose(1, 2)
        matmul_43 = None
        x_214 = transpose_44.reshape(1, 196, 384)
        transpose_44 = None
        x_215 = torch._C._nn.linear(
            x_214,
            l_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_bias_,
        )
        x_214 = l_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_216 = torch.nn.functional.dropout(x_215, 0.0, False, False)
        x_215 = None
        mul_64 = l_self_modules_blocks_modules_21_parameters_gamma_1_ * x_216
        l_self_modules_blocks_modules_21_parameters_gamma_1_ = x_216 = None
        x_217 = x_213 + mul_64
        x_213 = mul_64 = None
        layer_norm_43 = torch.nn.functional.layer_norm(
            x_217,
            (384,),
            l_self_modules_blocks_modules_21_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_21_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_21_modules_norm2_parameters_bias_
        ) = None
        x_218 = torch._C._nn.linear(
            layer_norm_43,
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_43 = (
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_219 = torch._C._nn.gelu(x_218, approximate="none")
        x_218 = None
        x_220 = torch.nn.functional.dropout(x_219, 0.0, False, False)
        x_219 = None
        x_221 = torch._C._nn.linear(
            x_220,
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_220 = (
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_222 = torch.nn.functional.dropout(x_221, 0.0, False, False)
        x_221 = None
        mul_65 = l_self_modules_blocks_modules_21_parameters_gamma_2_ * x_222
        l_self_modules_blocks_modules_21_parameters_gamma_2_ = x_222 = None
        x_223 = x_217 + mul_65
        x_217 = mul_65 = None
        layer_norm_44 = torch.nn.functional.layer_norm(
            x_223,
            (384,),
            l_self_modules_blocks_modules_22_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_22_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_22_modules_norm1_parameters_bias_
        ) = None
        linear_132 = torch._C._nn.linear(
            layer_norm_44,
            l_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_44 = (
            l_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_44 = linear_132.reshape(1, 196, 3, 8, 48)
        linear_132 = None
        qkv_22 = reshape_44.permute(2, 0, 3, 1, 4)
        reshape_44 = None
        getitem_70 = qkv_22[0]
        q_22 = getitem_70 * 0.14433756729740643
        getitem_70 = None
        k_22 = qkv_22[1]
        v_22 = qkv_22[2]
        qkv_22 = None
        transpose_45 = k_22.transpose(-2, -1)
        k_22 = None
        attn_110 = q_22 @ transpose_45
        q_22 = transpose_45 = None
        permute_111 = attn_110.permute(0, 2, 3, 1)
        attn_110 = None
        linear_133 = torch._C._nn.linear(
            permute_111,
            l_self_modules_blocks_modules_22_modules_attn_modules_proj_l_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_attn_modules_proj_l_parameters_bias_,
        )
        permute_111 = l_self_modules_blocks_modules_22_modules_attn_modules_proj_l_parameters_weight_ = l_self_modules_blocks_modules_22_modules_attn_modules_proj_l_parameters_bias_ = (None)
        attn_111 = linear_133.permute(0, 3, 1, 2)
        linear_133 = None
        attn_112 = attn_111.softmax(dim=-1)
        attn_111 = None
        permute_113 = attn_112.permute(0, 2, 3, 1)
        attn_112 = None
        linear_134 = torch._C._nn.linear(
            permute_113,
            l_self_modules_blocks_modules_22_modules_attn_modules_proj_w_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_attn_modules_proj_w_parameters_bias_,
        )
        permute_113 = l_self_modules_blocks_modules_22_modules_attn_modules_proj_w_parameters_weight_ = l_self_modules_blocks_modules_22_modules_attn_modules_proj_w_parameters_bias_ = (None)
        attn_113 = linear_134.permute(0, 3, 1, 2)
        linear_134 = None
        attn_114 = torch.nn.functional.dropout(attn_113, 0.0, False, False)
        attn_113 = None
        matmul_45 = attn_114 @ v_22
        attn_114 = v_22 = None
        transpose_46 = matmul_45.transpose(1, 2)
        matmul_45 = None
        x_224 = transpose_46.reshape(1, 196, 384)
        transpose_46 = None
        x_225 = torch._C._nn.linear(
            x_224,
            l_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_bias_,
        )
        x_224 = l_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_226 = torch.nn.functional.dropout(x_225, 0.0, False, False)
        x_225 = None
        mul_67 = l_self_modules_blocks_modules_22_parameters_gamma_1_ * x_226
        l_self_modules_blocks_modules_22_parameters_gamma_1_ = x_226 = None
        x_227 = x_223 + mul_67
        x_223 = mul_67 = None
        layer_norm_45 = torch.nn.functional.layer_norm(
            x_227,
            (384,),
            l_self_modules_blocks_modules_22_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_22_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_22_modules_norm2_parameters_bias_
        ) = None
        x_228 = torch._C._nn.linear(
            layer_norm_45,
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_45 = (
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_229 = torch._C._nn.gelu(x_228, approximate="none")
        x_228 = None
        x_230 = torch.nn.functional.dropout(x_229, 0.0, False, False)
        x_229 = None
        x_231 = torch._C._nn.linear(
            x_230,
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_230 = (
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_232 = torch.nn.functional.dropout(x_231, 0.0, False, False)
        x_231 = None
        mul_68 = l_self_modules_blocks_modules_22_parameters_gamma_2_ * x_232
        l_self_modules_blocks_modules_22_parameters_gamma_2_ = x_232 = None
        x_233 = x_227 + mul_68
        x_227 = mul_68 = None
        layer_norm_46 = torch.nn.functional.layer_norm(
            x_233,
            (384,),
            l_self_modules_blocks_modules_23_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_23_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_23_modules_norm1_parameters_bias_
        ) = None
        linear_138 = torch._C._nn.linear(
            layer_norm_46,
            l_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_bias_,
        )
        layer_norm_46 = (
            l_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_46 = linear_138.reshape(1, 196, 3, 8, 48)
        linear_138 = None
        qkv_23 = reshape_46.permute(2, 0, 3, 1, 4)
        reshape_46 = None
        getitem_73 = qkv_23[0]
        q_23 = getitem_73 * 0.14433756729740643
        getitem_73 = None
        k_23 = qkv_23[1]
        v_23 = qkv_23[2]
        qkv_23 = None
        transpose_47 = k_23.transpose(-2, -1)
        k_23 = None
        attn_115 = q_23 @ transpose_47
        q_23 = transpose_47 = None
        permute_116 = attn_115.permute(0, 2, 3, 1)
        attn_115 = None
        linear_139 = torch._C._nn.linear(
            permute_116,
            l_self_modules_blocks_modules_23_modules_attn_modules_proj_l_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_attn_modules_proj_l_parameters_bias_,
        )
        permute_116 = l_self_modules_blocks_modules_23_modules_attn_modules_proj_l_parameters_weight_ = l_self_modules_blocks_modules_23_modules_attn_modules_proj_l_parameters_bias_ = (None)
        attn_116 = linear_139.permute(0, 3, 1, 2)
        linear_139 = None
        attn_117 = attn_116.softmax(dim=-1)
        attn_116 = None
        permute_118 = attn_117.permute(0, 2, 3, 1)
        attn_117 = None
        linear_140 = torch._C._nn.linear(
            permute_118,
            l_self_modules_blocks_modules_23_modules_attn_modules_proj_w_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_attn_modules_proj_w_parameters_bias_,
        )
        permute_118 = l_self_modules_blocks_modules_23_modules_attn_modules_proj_w_parameters_weight_ = l_self_modules_blocks_modules_23_modules_attn_modules_proj_w_parameters_bias_ = (None)
        attn_118 = linear_140.permute(0, 3, 1, 2)
        linear_140 = None
        attn_119 = torch.nn.functional.dropout(attn_118, 0.0, False, False)
        attn_118 = None
        matmul_47 = attn_119 @ v_23
        attn_119 = v_23 = None
        transpose_48 = matmul_47.transpose(1, 2)
        matmul_47 = None
        x_234 = transpose_48.reshape(1, 196, 384)
        transpose_48 = None
        x_235 = torch._C._nn.linear(
            x_234,
            l_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_bias_,
        )
        x_234 = l_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_236 = torch.nn.functional.dropout(x_235, 0.0, False, False)
        x_235 = None
        mul_70 = l_self_modules_blocks_modules_23_parameters_gamma_1_ * x_236
        l_self_modules_blocks_modules_23_parameters_gamma_1_ = x_236 = None
        x_237 = x_233 + mul_70
        x_233 = mul_70 = None
        layer_norm_47 = torch.nn.functional.layer_norm(
            x_237,
            (384,),
            l_self_modules_blocks_modules_23_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_23_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_23_modules_norm2_parameters_bias_
        ) = None
        x_238 = torch._C._nn.linear(
            layer_norm_47,
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_47 = (
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_239 = torch._C._nn.gelu(x_238, approximate="none")
        x_238 = None
        x_240 = torch.nn.functional.dropout(x_239, 0.0, False, False)
        x_239 = None
        x_241 = torch._C._nn.linear(
            x_240,
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_240 = (
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_242 = torch.nn.functional.dropout(x_241, 0.0, False, False)
        x_241 = None
        mul_71 = l_self_modules_blocks_modules_23_parameters_gamma_2_ * x_242
        l_self_modules_blocks_modules_23_parameters_gamma_2_ = x_242 = None
        x_243 = x_237 + mul_71
        x_237 = mul_71 = None
        cls_tokens = l_self_parameters_cls_token_.expand(1, -1, -1)
        l_self_parameters_cls_token_ = None
        u = torch.cat((cls_tokens, x_243), dim=1)
        layer_norm_48 = torch.nn.functional.layer_norm(
            u,
            (384,),
            l_self_modules_blocks_token_only_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_blocks_token_only_modules_0_modules_norm1_parameters_bias_,
            1e-06,
        )
        u = (
            l_self_modules_blocks_token_only_modules_0_modules_norm1_parameters_weight_
        ) = (
            l_self_modules_blocks_token_only_modules_0_modules_norm1_parameters_bias_
        ) = None
        getitem_76 = layer_norm_48[(slice(None, None, None), 0)]
        linear_144 = torch._C._nn.linear(
            getitem_76,
            l_self_modules_blocks_token_only_modules_0_modules_attn_modules_q_parameters_weight_,
            l_self_modules_blocks_token_only_modules_0_modules_attn_modules_q_parameters_bias_,
        )
        getitem_76 = l_self_modules_blocks_token_only_modules_0_modules_attn_modules_q_parameters_weight_ = l_self_modules_blocks_token_only_modules_0_modules_attn_modules_q_parameters_bias_ = (None)
        unsqueeze = linear_144.unsqueeze(1)
        linear_144 = None
        reshape_48 = unsqueeze.reshape(1, 1, 8, 48)
        unsqueeze = None
        q_24 = reshape_48.permute(0, 2, 1, 3)
        reshape_48 = None
        linear_145 = torch._C._nn.linear(
            layer_norm_48,
            l_self_modules_blocks_token_only_modules_0_modules_attn_modules_k_parameters_weight_,
            l_self_modules_blocks_token_only_modules_0_modules_attn_modules_k_parameters_bias_,
        )
        l_self_modules_blocks_token_only_modules_0_modules_attn_modules_k_parameters_weight_ = l_self_modules_blocks_token_only_modules_0_modules_attn_modules_k_parameters_bias_ = (None)
        reshape_49 = linear_145.reshape(1, 197, 8, 48)
        linear_145 = None
        k_24 = reshape_49.permute(0, 2, 1, 3)
        reshape_49 = None
        linear_146 = torch._C._nn.linear(
            layer_norm_48,
            l_self_modules_blocks_token_only_modules_0_modules_attn_modules_v_parameters_weight_,
            l_self_modules_blocks_token_only_modules_0_modules_attn_modules_v_parameters_bias_,
        )
        layer_norm_48 = l_self_modules_blocks_token_only_modules_0_modules_attn_modules_v_parameters_weight_ = l_self_modules_blocks_token_only_modules_0_modules_attn_modules_v_parameters_bias_ = (None)
        reshape_50 = linear_146.reshape(1, 197, 8, 48)
        linear_146 = None
        v_24 = reshape_50.permute(0, 2, 1, 3)
        reshape_50 = None
        x_cls = torch._C._nn.scaled_dot_product_attention(
            q_24, k_24, v_24, dropout_p=0.0
        )
        q_24 = k_24 = v_24 = None
        transpose_49 = x_cls.transpose(1, 2)
        x_cls = None
        x_cls_1 = transpose_49.reshape(1, 1, 384)
        transpose_49 = None
        x_cls_2 = torch._C._nn.linear(
            x_cls_1,
            l_self_modules_blocks_token_only_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_token_only_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_cls_1 = l_self_modules_blocks_token_only_modules_0_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_token_only_modules_0_modules_attn_modules_proj_parameters_bias_ = (None)
        x_cls_3 = torch.nn.functional.dropout(x_cls_2, 0.0, False, False)
        x_cls_2 = None
        mul_72 = (
            l_self_modules_blocks_token_only_modules_0_parameters_gamma_1_ * x_cls_3
        )
        l_self_modules_blocks_token_only_modules_0_parameters_gamma_1_ = x_cls_3 = None
        x_cls_4 = cls_tokens + mul_72
        cls_tokens = mul_72 = None
        layer_norm_49 = torch.nn.functional.layer_norm(
            x_cls_4,
            (384,),
            l_self_modules_blocks_token_only_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_blocks_token_only_modules_0_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_token_only_modules_0_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_token_only_modules_0_modules_norm2_parameters_bias_
        ) = None
        x_244 = torch._C._nn.linear(
            layer_norm_49,
            l_self_modules_blocks_token_only_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_token_only_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_49 = l_self_modules_blocks_token_only_modules_0_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_blocks_token_only_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_245 = torch._C._nn.gelu(x_244, approximate="none")
        x_244 = None
        x_246 = torch.nn.functional.dropout(x_245, 0.0, False, False)
        x_245 = None
        x_247 = torch._C._nn.linear(
            x_246,
            l_self_modules_blocks_token_only_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_token_only_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_246 = l_self_modules_blocks_token_only_modules_0_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_blocks_token_only_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_248 = torch.nn.functional.dropout(x_247, 0.0, False, False)
        x_247 = None
        mul_73 = l_self_modules_blocks_token_only_modules_0_parameters_gamma_2_ * x_248
        l_self_modules_blocks_token_only_modules_0_parameters_gamma_2_ = x_248 = None
        x_cls_5 = x_cls_4 + mul_73
        x_cls_4 = mul_73 = None
        u_1 = torch.cat((x_cls_5, x_243), dim=1)
        layer_norm_50 = torch.nn.functional.layer_norm(
            u_1,
            (384,),
            l_self_modules_blocks_token_only_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_blocks_token_only_modules_1_modules_norm1_parameters_bias_,
            1e-06,
        )
        u_1 = (
            l_self_modules_blocks_token_only_modules_1_modules_norm1_parameters_weight_
        ) = (
            l_self_modules_blocks_token_only_modules_1_modules_norm1_parameters_bias_
        ) = None
        getitem_77 = layer_norm_50[(slice(None, None, None), 0)]
        linear_150 = torch._C._nn.linear(
            getitem_77,
            l_self_modules_blocks_token_only_modules_1_modules_attn_modules_q_parameters_weight_,
            l_self_modules_blocks_token_only_modules_1_modules_attn_modules_q_parameters_bias_,
        )
        getitem_77 = l_self_modules_blocks_token_only_modules_1_modules_attn_modules_q_parameters_weight_ = l_self_modules_blocks_token_only_modules_1_modules_attn_modules_q_parameters_bias_ = (None)
        unsqueeze_1 = linear_150.unsqueeze(1)
        linear_150 = None
        reshape_52 = unsqueeze_1.reshape(1, 1, 8, 48)
        unsqueeze_1 = None
        q_25 = reshape_52.permute(0, 2, 1, 3)
        reshape_52 = None
        linear_151 = torch._C._nn.linear(
            layer_norm_50,
            l_self_modules_blocks_token_only_modules_1_modules_attn_modules_k_parameters_weight_,
            l_self_modules_blocks_token_only_modules_1_modules_attn_modules_k_parameters_bias_,
        )
        l_self_modules_blocks_token_only_modules_1_modules_attn_modules_k_parameters_weight_ = l_self_modules_blocks_token_only_modules_1_modules_attn_modules_k_parameters_bias_ = (None)
        reshape_53 = linear_151.reshape(1, 197, 8, 48)
        linear_151 = None
        k_25 = reshape_53.permute(0, 2, 1, 3)
        reshape_53 = None
        linear_152 = torch._C._nn.linear(
            layer_norm_50,
            l_self_modules_blocks_token_only_modules_1_modules_attn_modules_v_parameters_weight_,
            l_self_modules_blocks_token_only_modules_1_modules_attn_modules_v_parameters_bias_,
        )
        layer_norm_50 = l_self_modules_blocks_token_only_modules_1_modules_attn_modules_v_parameters_weight_ = l_self_modules_blocks_token_only_modules_1_modules_attn_modules_v_parameters_bias_ = (None)
        reshape_54 = linear_152.reshape(1, 197, 8, 48)
        linear_152 = None
        v_25 = reshape_54.permute(0, 2, 1, 3)
        reshape_54 = None
        x_cls_6 = torch._C._nn.scaled_dot_product_attention(
            q_25, k_25, v_25, dropout_p=0.0
        )
        q_25 = k_25 = v_25 = None
        transpose_50 = x_cls_6.transpose(1, 2)
        x_cls_6 = None
        x_cls_7 = transpose_50.reshape(1, 1, 384)
        transpose_50 = None
        x_cls_8 = torch._C._nn.linear(
            x_cls_7,
            l_self_modules_blocks_token_only_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_token_only_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_cls_7 = l_self_modules_blocks_token_only_modules_1_modules_attn_modules_proj_parameters_weight_ = l_self_modules_blocks_token_only_modules_1_modules_attn_modules_proj_parameters_bias_ = (None)
        x_cls_9 = torch.nn.functional.dropout(x_cls_8, 0.0, False, False)
        x_cls_8 = None
        mul_74 = (
            l_self_modules_blocks_token_only_modules_1_parameters_gamma_1_ * x_cls_9
        )
        l_self_modules_blocks_token_only_modules_1_parameters_gamma_1_ = x_cls_9 = None
        x_cls_10 = x_cls_5 + mul_74
        x_cls_5 = mul_74 = None
        layer_norm_51 = torch.nn.functional.layer_norm(
            x_cls_10,
            (384,),
            l_self_modules_blocks_token_only_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_blocks_token_only_modules_1_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_token_only_modules_1_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_token_only_modules_1_modules_norm2_parameters_bias_
        ) = None
        x_249 = torch._C._nn.linear(
            layer_norm_51,
            l_self_modules_blocks_token_only_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_token_only_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_51 = l_self_modules_blocks_token_only_modules_1_modules_mlp_modules_fc1_parameters_weight_ = l_self_modules_blocks_token_only_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (None)
        x_250 = torch._C._nn.gelu(x_249, approximate="none")
        x_249 = None
        x_251 = torch.nn.functional.dropout(x_250, 0.0, False, False)
        x_250 = None
        x_252 = torch._C._nn.linear(
            x_251,
            l_self_modules_blocks_token_only_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_token_only_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_251 = l_self_modules_blocks_token_only_modules_1_modules_mlp_modules_fc2_parameters_weight_ = l_self_modules_blocks_token_only_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (None)
        x_253 = torch.nn.functional.dropout(x_252, 0.0, False, False)
        x_252 = None
        mul_75 = l_self_modules_blocks_token_only_modules_1_parameters_gamma_2_ * x_253
        l_self_modules_blocks_token_only_modules_1_parameters_gamma_2_ = x_253 = None
        x_cls_11 = x_cls_10 + mul_75
        x_cls_10 = mul_75 = None
        x_254 = torch.cat((x_cls_11, x_243), dim=1)
        x_cls_11 = x_243 = None
        x_255 = torch.nn.functional.layer_norm(
            x_254,
            (384,),
            l_self_modules_norm_parameters_weight_,
            l_self_modules_norm_parameters_bias_,
            1e-06,
        )
        x_254 = (
            l_self_modules_norm_parameters_weight_
        ) = l_self_modules_norm_parameters_bias_ = None
        x_256 = x_255[(slice(None, None, None), 0)]
        x_255 = None
        x_257 = torch.nn.functional.dropout(x_256, 0.0, False, False)
        x_256 = None
        x_258 = torch._C._nn.linear(
            x_257,
            l_self_modules_head_parameters_weight_,
            l_self_modules_head_parameters_bias_,
        )
        x_257 = (
            l_self_modules_head_parameters_weight_
        ) = l_self_modules_head_parameters_bias_ = None
        return (x_258,)
