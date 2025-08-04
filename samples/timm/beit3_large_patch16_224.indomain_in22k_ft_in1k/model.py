import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_x_: torch.Tensor,
        L_self_modules_patch_embed_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_patch_embed_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_parameters_pos_embed_: torch.nn.parameter.Parameter,
        L_self_parameters_cls_token_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_mlp_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_mlp_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_mlp_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_mlp_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_mlp_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_mlp_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_mlp_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_mlp_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_mlp_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_mlp_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_mlp_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_mlp_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_mlp_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_mlp_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_mlp_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_mlp_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_mlp_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_mlp_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_mlp_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_mlp_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_mlp_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_mlp_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_mlp_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_mlp_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_mlp_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_mlp_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_mlp_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_mlp_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_mlp_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_mlp_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_mlp_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_mlp_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_mlp_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_mlp_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_mlp_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_mlp_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_mlp_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_mlp_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_mlp_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_mlp_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_mlp_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_mlp_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_mlp_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_mlp_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_mlp_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_mlp_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_num_prefix_tokens: torch.SymInt,
        L_self_modules_fc_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_fc_norm_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_parameters_cls_token_ = L_self_parameters_cls_token_
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
        l_self_modules_blocks_modules_0_modules_attn_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_attn_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_attn_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_0_modules_attn_modules_norm_parameters_bias_
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
        l_self_modules_blocks_modules_0_modules_mlp_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_mlp_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_mlp_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_0_modules_mlp_modules_norm_parameters_bias_
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
        l_self_modules_blocks_modules_1_modules_attn_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_attn_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_attn_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_attn_modules_norm_parameters_bias_
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
        l_self_modules_blocks_modules_1_modules_mlp_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_mlp_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_mlp_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_mlp_modules_norm_parameters_bias_
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
        l_self_modules_blocks_modules_2_modules_attn_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_attn_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_attn_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_attn_modules_norm_parameters_bias_
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
        l_self_modules_blocks_modules_2_modules_mlp_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_mlp_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_mlp_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_mlp_modules_norm_parameters_bias_
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
        l_self_modules_blocks_modules_3_modules_attn_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_attn_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_attn_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_attn_modules_norm_parameters_bias_
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
        l_self_modules_blocks_modules_3_modules_mlp_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_mlp_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_mlp_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_mlp_modules_norm_parameters_bias_
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
        l_self_modules_blocks_modules_4_modules_attn_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_attn_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_attn_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_attn_modules_norm_parameters_bias_
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
        l_self_modules_blocks_modules_4_modules_mlp_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_mlp_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_mlp_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_mlp_modules_norm_parameters_bias_
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
        l_self_modules_blocks_modules_5_modules_attn_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_attn_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_attn_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_attn_modules_norm_parameters_bias_
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
        l_self_modules_blocks_modules_5_modules_mlp_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_mlp_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_mlp_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_mlp_modules_norm_parameters_bias_
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
        l_self_modules_blocks_modules_6_modules_attn_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_attn_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_attn_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_6_modules_attn_modules_norm_parameters_bias_
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
        l_self_modules_blocks_modules_6_modules_mlp_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_mlp_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_mlp_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_6_modules_mlp_modules_norm_parameters_bias_
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
        l_self_modules_blocks_modules_7_modules_attn_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_7_modules_attn_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_7_modules_attn_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_7_modules_attn_modules_norm_parameters_bias_
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
        l_self_modules_blocks_modules_7_modules_mlp_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_7_modules_mlp_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_7_modules_mlp_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_7_modules_mlp_modules_norm_parameters_bias_
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
        l_self_modules_blocks_modules_8_modules_attn_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_8_modules_attn_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_8_modules_attn_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_8_modules_attn_modules_norm_parameters_bias_
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
        l_self_modules_blocks_modules_8_modules_mlp_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_8_modules_mlp_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_8_modules_mlp_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_8_modules_mlp_modules_norm_parameters_bias_
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
        l_self_modules_blocks_modules_9_modules_attn_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_9_modules_attn_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_9_modules_attn_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_9_modules_attn_modules_norm_parameters_bias_
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
        l_self_modules_blocks_modules_9_modules_mlp_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_9_modules_mlp_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_9_modules_mlp_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_9_modules_mlp_modules_norm_parameters_bias_
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
        l_self_modules_blocks_modules_10_modules_attn_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_10_modules_attn_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_10_modules_attn_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_10_modules_attn_modules_norm_parameters_bias_
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
        l_self_modules_blocks_modules_10_modules_mlp_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_10_modules_mlp_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_10_modules_mlp_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_10_modules_mlp_modules_norm_parameters_bias_
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
        l_self_modules_blocks_modules_11_modules_attn_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_11_modules_attn_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_11_modules_attn_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_11_modules_attn_modules_norm_parameters_bias_
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
        l_self_modules_blocks_modules_11_modules_mlp_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_11_modules_mlp_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_11_modules_mlp_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_11_modules_mlp_modules_norm_parameters_bias_
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
        l_self_modules_blocks_modules_12_modules_attn_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_12_modules_attn_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_12_modules_attn_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_12_modules_attn_modules_norm_parameters_bias_
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
        l_self_modules_blocks_modules_12_modules_mlp_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_12_modules_mlp_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_12_modules_mlp_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_12_modules_mlp_modules_norm_parameters_bias_
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
        l_self_modules_blocks_modules_13_modules_attn_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_13_modules_attn_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_13_modules_attn_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_13_modules_attn_modules_norm_parameters_bias_
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
        l_self_modules_blocks_modules_13_modules_mlp_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_13_modules_mlp_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_13_modules_mlp_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_13_modules_mlp_modules_norm_parameters_bias_
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
        l_self_modules_blocks_modules_14_modules_attn_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_14_modules_attn_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_14_modules_attn_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_14_modules_attn_modules_norm_parameters_bias_
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
        l_self_modules_blocks_modules_14_modules_mlp_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_14_modules_mlp_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_14_modules_mlp_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_14_modules_mlp_modules_norm_parameters_bias_
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
        l_self_modules_blocks_modules_15_modules_attn_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_15_modules_attn_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_15_modules_attn_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_15_modules_attn_modules_norm_parameters_bias_
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
        l_self_modules_blocks_modules_15_modules_mlp_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_15_modules_mlp_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_15_modules_mlp_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_15_modules_mlp_modules_norm_parameters_bias_
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
        l_self_modules_blocks_modules_16_modules_attn_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_16_modules_attn_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_16_modules_attn_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_16_modules_attn_modules_norm_parameters_bias_
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
        l_self_modules_blocks_modules_16_modules_mlp_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_16_modules_mlp_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_16_modules_mlp_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_16_modules_mlp_modules_norm_parameters_bias_
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
        l_self_modules_blocks_modules_17_modules_attn_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_17_modules_attn_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_17_modules_attn_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_17_modules_attn_modules_norm_parameters_bias_
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
        l_self_modules_blocks_modules_17_modules_mlp_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_17_modules_mlp_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_17_modules_mlp_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_17_modules_mlp_modules_norm_parameters_bias_
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
        l_self_modules_blocks_modules_18_modules_attn_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_18_modules_attn_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_18_modules_attn_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_18_modules_attn_modules_norm_parameters_bias_
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
        l_self_modules_blocks_modules_18_modules_mlp_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_18_modules_mlp_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_18_modules_mlp_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_18_modules_mlp_modules_norm_parameters_bias_
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
        l_self_modules_blocks_modules_19_modules_attn_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_19_modules_attn_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_19_modules_attn_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_19_modules_attn_modules_norm_parameters_bias_
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
        l_self_modules_blocks_modules_19_modules_mlp_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_19_modules_mlp_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_19_modules_mlp_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_19_modules_mlp_modules_norm_parameters_bias_
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
        l_self_modules_blocks_modules_20_modules_attn_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_20_modules_attn_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_20_modules_attn_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_20_modules_attn_modules_norm_parameters_bias_
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
        l_self_modules_blocks_modules_20_modules_mlp_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_20_modules_mlp_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_20_modules_mlp_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_20_modules_mlp_modules_norm_parameters_bias_
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
        l_self_modules_blocks_modules_21_modules_attn_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_21_modules_attn_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_21_modules_attn_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_21_modules_attn_modules_norm_parameters_bias_
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
        l_self_modules_blocks_modules_21_modules_mlp_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_21_modules_mlp_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_21_modules_mlp_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_21_modules_mlp_modules_norm_parameters_bias_
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
        l_self_modules_blocks_modules_22_modules_attn_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_22_modules_attn_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_22_modules_attn_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_22_modules_attn_modules_norm_parameters_bias_
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
        l_self_modules_blocks_modules_22_modules_mlp_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_22_modules_mlp_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_22_modules_mlp_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_22_modules_mlp_modules_norm_parameters_bias_
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
        l_self_modules_blocks_modules_23_modules_attn_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_23_modules_attn_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_23_modules_attn_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_23_modules_attn_modules_norm_parameters_bias_
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
        l_self_modules_blocks_modules_23_modules_mlp_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_23_modules_mlp_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_23_modules_mlp_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_23_modules_mlp_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_num_prefix_tokens = L_self_num_prefix_tokens
        l_self_modules_fc_norm_parameters_weight_ = (
            L_self_modules_fc_norm_parameters_weight_
        )
        l_self_modules_fc_norm_parameters_bias_ = (
            L_self_modules_fc_norm_parameters_bias_
        )
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
        x_3 = x_2 + l_self_parameters_pos_embed_
        x_2 = l_self_parameters_pos_embed_ = None
        x_4 = torch.nn.functional.dropout(x_3, 0.0, False, False)
        x_3 = None
        x_5 = torch.nn.functional.layer_norm(
            x_4,
            (1024,),
            l_self_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_0_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_0_modules_norm1_parameters_bias_
        ) = None
        linear = torch._C._nn.linear(
            x_5,
            l_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_,
        )
        x_5 = (
            l_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape = linear.reshape(1, 197, 3, 16, 64)
        linear = None
        qkv = reshape.permute(2, 0, 3, 1, 4)
        reshape = None
        unbind = qkv.unbind(0)
        qkv = None
        q = unbind[0]
        k = unbind[1]
        v = unbind[2]
        unbind = None
        x_6 = torch._C._nn.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0
        )
        q = k = v = None
        transpose_1 = x_6.transpose(1, 2)
        x_6 = None
        x_7 = transpose_1.reshape(1, 197, 1024)
        transpose_1 = None
        x_8 = torch.nn.functional.layer_norm(
            x_7,
            (1024,),
            l_self_modules_blocks_modules_0_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_7 = (
            l_self_modules_blocks_modules_0_modules_attn_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_attn_modules_norm_parameters_bias_
        ) = None
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
        x_11 = x_4 + x_10
        x_4 = x_10 = None
        x_12 = torch.nn.functional.layer_norm(
            x_11,
            (1024,),
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
        x_14 = torch._C._nn.gelu(x_13, approximate="none")
        x_13 = None
        x_15 = torch.nn.functional.dropout(x_14, 0.0, False, False)
        x_14 = None
        x_16 = torch.nn.functional.layer_norm(
            x_15,
            (4096,),
            l_self_modules_blocks_modules_0_modules_mlp_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_mlp_modules_norm_parameters_bias_,
            1e-05,
        )
        x_15 = (
            l_self_modules_blocks_modules_0_modules_mlp_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_mlp_modules_norm_parameters_bias_
        ) = None
        x_17 = torch._C._nn.linear(
            x_16,
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_16 = (
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_18 = torch.nn.functional.dropout(x_17, 0.0, False, False)
        x_17 = None
        x_19 = x_11 + x_18
        x_11 = x_18 = None
        x_20 = torch.nn.functional.layer_norm(
            x_19,
            (1024,),
            l_self_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_1_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_1_modules_norm1_parameters_bias_
        ) = None
        linear_4 = torch._C._nn.linear(
            x_20,
            l_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_,
        )
        x_20 = (
            l_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_2 = linear_4.reshape(1, 197, 3, 16, 64)
        linear_4 = None
        qkv_1 = reshape_2.permute(2, 0, 3, 1, 4)
        reshape_2 = None
        unbind_1 = qkv_1.unbind(0)
        qkv_1 = None
        q_1 = unbind_1[0]
        k_1 = unbind_1[1]
        v_1 = unbind_1[2]
        unbind_1 = None
        x_21 = torch._C._nn.scaled_dot_product_attention(
            q_1, k_1, v_1, attn_mask=None, dropout_p=0.0
        )
        q_1 = k_1 = v_1 = None
        transpose_2 = x_21.transpose(1, 2)
        x_21 = None
        x_22 = transpose_2.reshape(1, 197, 1024)
        transpose_2 = None
        x_23 = torch.nn.functional.layer_norm(
            x_22,
            (1024,),
            l_self_modules_blocks_modules_1_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_22 = (
            l_self_modules_blocks_modules_1_modules_attn_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_attn_modules_norm_parameters_bias_
        ) = None
        x_24 = torch._C._nn.linear(
            x_23,
            l_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_23 = (
            l_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_25 = torch.nn.functional.dropout(x_24, 0.0, False, False)
        x_24 = None
        x_26 = x_19 + x_25
        x_19 = x_25 = None
        x_27 = torch.nn.functional.layer_norm(
            x_26,
            (1024,),
            l_self_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_1_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_1_modules_norm2_parameters_bias_
        ) = None
        x_28 = torch._C._nn.linear(
            x_27,
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_27 = (
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_29 = torch._C._nn.gelu(x_28, approximate="none")
        x_28 = None
        x_30 = torch.nn.functional.dropout(x_29, 0.0, False, False)
        x_29 = None
        x_31 = torch.nn.functional.layer_norm(
            x_30,
            (4096,),
            l_self_modules_blocks_modules_1_modules_mlp_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_mlp_modules_norm_parameters_bias_,
            1e-05,
        )
        x_30 = (
            l_self_modules_blocks_modules_1_modules_mlp_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_mlp_modules_norm_parameters_bias_
        ) = None
        x_32 = torch._C._nn.linear(
            x_31,
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_31 = (
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_33 = torch.nn.functional.dropout(x_32, 0.0, False, False)
        x_32 = None
        x_34 = x_26 + x_33
        x_26 = x_33 = None
        x_35 = torch.nn.functional.layer_norm(
            x_34,
            (1024,),
            l_self_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_2_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_2_modules_norm1_parameters_bias_
        ) = None
        linear_8 = torch._C._nn.linear(
            x_35,
            l_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_,
        )
        x_35 = (
            l_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_4 = linear_8.reshape(1, 197, 3, 16, 64)
        linear_8 = None
        qkv_2 = reshape_4.permute(2, 0, 3, 1, 4)
        reshape_4 = None
        unbind_2 = qkv_2.unbind(0)
        qkv_2 = None
        q_2 = unbind_2[0]
        k_2 = unbind_2[1]
        v_2 = unbind_2[2]
        unbind_2 = None
        x_36 = torch._C._nn.scaled_dot_product_attention(
            q_2, k_2, v_2, attn_mask=None, dropout_p=0.0
        )
        q_2 = k_2 = v_2 = None
        transpose_3 = x_36.transpose(1, 2)
        x_36 = None
        x_37 = transpose_3.reshape(1, 197, 1024)
        transpose_3 = None
        x_38 = torch.nn.functional.layer_norm(
            x_37,
            (1024,),
            l_self_modules_blocks_modules_2_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_37 = (
            l_self_modules_blocks_modules_2_modules_attn_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_attn_modules_norm_parameters_bias_
        ) = None
        x_39 = torch._C._nn.linear(
            x_38,
            l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_,
        )
        x_38 = (
            l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_40 = torch.nn.functional.dropout(x_39, 0.0, False, False)
        x_39 = None
        x_41 = x_34 + x_40
        x_34 = x_40 = None
        x_42 = torch.nn.functional.layer_norm(
            x_41,
            (1024,),
            l_self_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_2_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_2_modules_norm2_parameters_bias_
        ) = None
        x_43 = torch._C._nn.linear(
            x_42,
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_42 = (
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_44 = torch._C._nn.gelu(x_43, approximate="none")
        x_43 = None
        x_45 = torch.nn.functional.dropout(x_44, 0.0, False, False)
        x_44 = None
        x_46 = torch.nn.functional.layer_norm(
            x_45,
            (4096,),
            l_self_modules_blocks_modules_2_modules_mlp_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_mlp_modules_norm_parameters_bias_,
            1e-05,
        )
        x_45 = (
            l_self_modules_blocks_modules_2_modules_mlp_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_mlp_modules_norm_parameters_bias_
        ) = None
        x_47 = torch._C._nn.linear(
            x_46,
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_46 = (
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_48 = torch.nn.functional.dropout(x_47, 0.0, False, False)
        x_47 = None
        x_49 = x_41 + x_48
        x_41 = x_48 = None
        x_50 = torch.nn.functional.layer_norm(
            x_49,
            (1024,),
            l_self_modules_blocks_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_3_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_3_modules_norm1_parameters_bias_
        ) = None
        linear_12 = torch._C._nn.linear(
            x_50,
            l_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_bias_,
        )
        x_50 = (
            l_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_6 = linear_12.reshape(1, 197, 3, 16, 64)
        linear_12 = None
        qkv_3 = reshape_6.permute(2, 0, 3, 1, 4)
        reshape_6 = None
        unbind_3 = qkv_3.unbind(0)
        qkv_3 = None
        q_3 = unbind_3[0]
        k_3 = unbind_3[1]
        v_3 = unbind_3[2]
        unbind_3 = None
        x_51 = torch._C._nn.scaled_dot_product_attention(
            q_3, k_3, v_3, attn_mask=None, dropout_p=0.0
        )
        q_3 = k_3 = v_3 = None
        transpose_4 = x_51.transpose(1, 2)
        x_51 = None
        x_52 = transpose_4.reshape(1, 197, 1024)
        transpose_4 = None
        x_53 = torch.nn.functional.layer_norm(
            x_52,
            (1024,),
            l_self_modules_blocks_modules_3_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_52 = (
            l_self_modules_blocks_modules_3_modules_attn_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_attn_modules_norm_parameters_bias_
        ) = None
        x_54 = torch._C._nn.linear(
            x_53,
            l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_,
        )
        x_53 = (
            l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_55 = torch.nn.functional.dropout(x_54, 0.0, False, False)
        x_54 = None
        x_56 = x_49 + x_55
        x_49 = x_55 = None
        x_57 = torch.nn.functional.layer_norm(
            x_56,
            (1024,),
            l_self_modules_blocks_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_3_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_3_modules_norm2_parameters_bias_
        ) = None
        x_58 = torch._C._nn.linear(
            x_57,
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_57 = (
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_59 = torch._C._nn.gelu(x_58, approximate="none")
        x_58 = None
        x_60 = torch.nn.functional.dropout(x_59, 0.0, False, False)
        x_59 = None
        x_61 = torch.nn.functional.layer_norm(
            x_60,
            (4096,),
            l_self_modules_blocks_modules_3_modules_mlp_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_mlp_modules_norm_parameters_bias_,
            1e-05,
        )
        x_60 = (
            l_self_modules_blocks_modules_3_modules_mlp_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_mlp_modules_norm_parameters_bias_
        ) = None
        x_62 = torch._C._nn.linear(
            x_61,
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_61 = (
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_63 = torch.nn.functional.dropout(x_62, 0.0, False, False)
        x_62 = None
        x_64 = x_56 + x_63
        x_56 = x_63 = None
        x_65 = torch.nn.functional.layer_norm(
            x_64,
            (1024,),
            l_self_modules_blocks_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_4_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_4_modules_norm1_parameters_bias_
        ) = None
        linear_16 = torch._C._nn.linear(
            x_65,
            l_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_bias_,
        )
        x_65 = (
            l_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_8 = linear_16.reshape(1, 197, 3, 16, 64)
        linear_16 = None
        qkv_4 = reshape_8.permute(2, 0, 3, 1, 4)
        reshape_8 = None
        unbind_4 = qkv_4.unbind(0)
        qkv_4 = None
        q_4 = unbind_4[0]
        k_4 = unbind_4[1]
        v_4 = unbind_4[2]
        unbind_4 = None
        x_66 = torch._C._nn.scaled_dot_product_attention(
            q_4, k_4, v_4, attn_mask=None, dropout_p=0.0
        )
        q_4 = k_4 = v_4 = None
        transpose_5 = x_66.transpose(1, 2)
        x_66 = None
        x_67 = transpose_5.reshape(1, 197, 1024)
        transpose_5 = None
        x_68 = torch.nn.functional.layer_norm(
            x_67,
            (1024,),
            l_self_modules_blocks_modules_4_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_67 = (
            l_self_modules_blocks_modules_4_modules_attn_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_attn_modules_norm_parameters_bias_
        ) = None
        x_69 = torch._C._nn.linear(
            x_68,
            l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_,
        )
        x_68 = (
            l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_70 = torch.nn.functional.dropout(x_69, 0.0, False, False)
        x_69 = None
        x_71 = x_64 + x_70
        x_64 = x_70 = None
        x_72 = torch.nn.functional.layer_norm(
            x_71,
            (1024,),
            l_self_modules_blocks_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_4_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_4_modules_norm2_parameters_bias_
        ) = None
        x_73 = torch._C._nn.linear(
            x_72,
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_72 = (
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_74 = torch._C._nn.gelu(x_73, approximate="none")
        x_73 = None
        x_75 = torch.nn.functional.dropout(x_74, 0.0, False, False)
        x_74 = None
        x_76 = torch.nn.functional.layer_norm(
            x_75,
            (4096,),
            l_self_modules_blocks_modules_4_modules_mlp_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_mlp_modules_norm_parameters_bias_,
            1e-05,
        )
        x_75 = (
            l_self_modules_blocks_modules_4_modules_mlp_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_mlp_modules_norm_parameters_bias_
        ) = None
        x_77 = torch._C._nn.linear(
            x_76,
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_76 = (
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_78 = torch.nn.functional.dropout(x_77, 0.0, False, False)
        x_77 = None
        x_79 = x_71 + x_78
        x_71 = x_78 = None
        x_80 = torch.nn.functional.layer_norm(
            x_79,
            (1024,),
            l_self_modules_blocks_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_5_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_5_modules_norm1_parameters_bias_
        ) = None
        linear_20 = torch._C._nn.linear(
            x_80,
            l_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_bias_,
        )
        x_80 = (
            l_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_10 = linear_20.reshape(1, 197, 3, 16, 64)
        linear_20 = None
        qkv_5 = reshape_10.permute(2, 0, 3, 1, 4)
        reshape_10 = None
        unbind_5 = qkv_5.unbind(0)
        qkv_5 = None
        q_5 = unbind_5[0]
        k_5 = unbind_5[1]
        v_5 = unbind_5[2]
        unbind_5 = None
        x_81 = torch._C._nn.scaled_dot_product_attention(
            q_5, k_5, v_5, attn_mask=None, dropout_p=0.0
        )
        q_5 = k_5 = v_5 = None
        transpose_6 = x_81.transpose(1, 2)
        x_81 = None
        x_82 = transpose_6.reshape(1, 197, 1024)
        transpose_6 = None
        x_83 = torch.nn.functional.layer_norm(
            x_82,
            (1024,),
            l_self_modules_blocks_modules_5_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_82 = (
            l_self_modules_blocks_modules_5_modules_attn_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_attn_modules_norm_parameters_bias_
        ) = None
        x_84 = torch._C._nn.linear(
            x_83,
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_,
        )
        x_83 = (
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_85 = torch.nn.functional.dropout(x_84, 0.0, False, False)
        x_84 = None
        x_86 = x_79 + x_85
        x_79 = x_85 = None
        x_87 = torch.nn.functional.layer_norm(
            x_86,
            (1024,),
            l_self_modules_blocks_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_5_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_5_modules_norm2_parameters_bias_
        ) = None
        x_88 = torch._C._nn.linear(
            x_87,
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_87 = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_89 = torch._C._nn.gelu(x_88, approximate="none")
        x_88 = None
        x_90 = torch.nn.functional.dropout(x_89, 0.0, False, False)
        x_89 = None
        x_91 = torch.nn.functional.layer_norm(
            x_90,
            (4096,),
            l_self_modules_blocks_modules_5_modules_mlp_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_mlp_modules_norm_parameters_bias_,
            1e-05,
        )
        x_90 = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_norm_parameters_bias_
        ) = None
        x_92 = torch._C._nn.linear(
            x_91,
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_91 = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_93 = torch.nn.functional.dropout(x_92, 0.0, False, False)
        x_92 = None
        x_94 = x_86 + x_93
        x_86 = x_93 = None
        x_95 = torch.nn.functional.layer_norm(
            x_94,
            (1024,),
            l_self_modules_blocks_modules_6_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_6_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_6_modules_norm1_parameters_bias_
        ) = None
        linear_24 = torch._C._nn.linear(
            x_95,
            l_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_bias_,
        )
        x_95 = (
            l_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_12 = linear_24.reshape(1, 197, 3, 16, 64)
        linear_24 = None
        qkv_6 = reshape_12.permute(2, 0, 3, 1, 4)
        reshape_12 = None
        unbind_6 = qkv_6.unbind(0)
        qkv_6 = None
        q_6 = unbind_6[0]
        k_6 = unbind_6[1]
        v_6 = unbind_6[2]
        unbind_6 = None
        x_96 = torch._C._nn.scaled_dot_product_attention(
            q_6, k_6, v_6, attn_mask=None, dropout_p=0.0
        )
        q_6 = k_6 = v_6 = None
        transpose_7 = x_96.transpose(1, 2)
        x_96 = None
        x_97 = transpose_7.reshape(1, 197, 1024)
        transpose_7 = None
        x_98 = torch.nn.functional.layer_norm(
            x_97,
            (1024,),
            l_self_modules_blocks_modules_6_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_97 = (
            l_self_modules_blocks_modules_6_modules_attn_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_attn_modules_norm_parameters_bias_
        ) = None
        x_99 = torch._C._nn.linear(
            x_98,
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_,
        )
        x_98 = (
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_100 = torch.nn.functional.dropout(x_99, 0.0, False, False)
        x_99 = None
        x_101 = x_94 + x_100
        x_94 = x_100 = None
        x_102 = torch.nn.functional.layer_norm(
            x_101,
            (1024,),
            l_self_modules_blocks_modules_6_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_6_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_6_modules_norm2_parameters_bias_
        ) = None
        x_103 = torch._C._nn.linear(
            x_102,
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_102 = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_104 = torch._C._nn.gelu(x_103, approximate="none")
        x_103 = None
        x_105 = torch.nn.functional.dropout(x_104, 0.0, False, False)
        x_104 = None
        x_106 = torch.nn.functional.layer_norm(
            x_105,
            (4096,),
            l_self_modules_blocks_modules_6_modules_mlp_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_mlp_modules_norm_parameters_bias_,
            1e-05,
        )
        x_105 = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_norm_parameters_bias_
        ) = None
        x_107 = torch._C._nn.linear(
            x_106,
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_106 = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_108 = torch.nn.functional.dropout(x_107, 0.0, False, False)
        x_107 = None
        x_109 = x_101 + x_108
        x_101 = x_108 = None
        x_110 = torch.nn.functional.layer_norm(
            x_109,
            (1024,),
            l_self_modules_blocks_modules_7_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_7_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_7_modules_norm1_parameters_bias_
        ) = None
        linear_28 = torch._C._nn.linear(
            x_110,
            l_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_bias_,
        )
        x_110 = (
            l_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_14 = linear_28.reshape(1, 197, 3, 16, 64)
        linear_28 = None
        qkv_7 = reshape_14.permute(2, 0, 3, 1, 4)
        reshape_14 = None
        unbind_7 = qkv_7.unbind(0)
        qkv_7 = None
        q_7 = unbind_7[0]
        k_7 = unbind_7[1]
        v_7 = unbind_7[2]
        unbind_7 = None
        x_111 = torch._C._nn.scaled_dot_product_attention(
            q_7, k_7, v_7, attn_mask=None, dropout_p=0.0
        )
        q_7 = k_7 = v_7 = None
        transpose_8 = x_111.transpose(1, 2)
        x_111 = None
        x_112 = transpose_8.reshape(1, 197, 1024)
        transpose_8 = None
        x_113 = torch.nn.functional.layer_norm(
            x_112,
            (1024,),
            l_self_modules_blocks_modules_7_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_112 = (
            l_self_modules_blocks_modules_7_modules_attn_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_7_modules_attn_modules_norm_parameters_bias_
        ) = None
        x_114 = torch._C._nn.linear(
            x_113,
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_,
        )
        x_113 = (
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_115 = torch.nn.functional.dropout(x_114, 0.0, False, False)
        x_114 = None
        x_116 = x_109 + x_115
        x_109 = x_115 = None
        x_117 = torch.nn.functional.layer_norm(
            x_116,
            (1024,),
            l_self_modules_blocks_modules_7_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_7_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_7_modules_norm2_parameters_bias_
        ) = None
        x_118 = torch._C._nn.linear(
            x_117,
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_117 = (
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_119 = torch._C._nn.gelu(x_118, approximate="none")
        x_118 = None
        x_120 = torch.nn.functional.dropout(x_119, 0.0, False, False)
        x_119 = None
        x_121 = torch.nn.functional.layer_norm(
            x_120,
            (4096,),
            l_self_modules_blocks_modules_7_modules_mlp_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_mlp_modules_norm_parameters_bias_,
            1e-05,
        )
        x_120 = (
            l_self_modules_blocks_modules_7_modules_mlp_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_7_modules_mlp_modules_norm_parameters_bias_
        ) = None
        x_122 = torch._C._nn.linear(
            x_121,
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_121 = (
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_123 = torch.nn.functional.dropout(x_122, 0.0, False, False)
        x_122 = None
        x_124 = x_116 + x_123
        x_116 = x_123 = None
        x_125 = torch.nn.functional.layer_norm(
            x_124,
            (1024,),
            l_self_modules_blocks_modules_8_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_8_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_8_modules_norm1_parameters_bias_
        ) = None
        linear_32 = torch._C._nn.linear(
            x_125,
            l_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_bias_,
        )
        x_125 = (
            l_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_16 = linear_32.reshape(1, 197, 3, 16, 64)
        linear_32 = None
        qkv_8 = reshape_16.permute(2, 0, 3, 1, 4)
        reshape_16 = None
        unbind_8 = qkv_8.unbind(0)
        qkv_8 = None
        q_8 = unbind_8[0]
        k_8 = unbind_8[1]
        v_8 = unbind_8[2]
        unbind_8 = None
        x_126 = torch._C._nn.scaled_dot_product_attention(
            q_8, k_8, v_8, attn_mask=None, dropout_p=0.0
        )
        q_8 = k_8 = v_8 = None
        transpose_9 = x_126.transpose(1, 2)
        x_126 = None
        x_127 = transpose_9.reshape(1, 197, 1024)
        transpose_9 = None
        x_128 = torch.nn.functional.layer_norm(
            x_127,
            (1024,),
            l_self_modules_blocks_modules_8_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_127 = (
            l_self_modules_blocks_modules_8_modules_attn_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_8_modules_attn_modules_norm_parameters_bias_
        ) = None
        x_129 = torch._C._nn.linear(
            x_128,
            l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_,
        )
        x_128 = (
            l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_130 = torch.nn.functional.dropout(x_129, 0.0, False, False)
        x_129 = None
        x_131 = x_124 + x_130
        x_124 = x_130 = None
        x_132 = torch.nn.functional.layer_norm(
            x_131,
            (1024,),
            l_self_modules_blocks_modules_8_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_8_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_8_modules_norm2_parameters_bias_
        ) = None
        x_133 = torch._C._nn.linear(
            x_132,
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_132 = (
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_134 = torch._C._nn.gelu(x_133, approximate="none")
        x_133 = None
        x_135 = torch.nn.functional.dropout(x_134, 0.0, False, False)
        x_134 = None
        x_136 = torch.nn.functional.layer_norm(
            x_135,
            (4096,),
            l_self_modules_blocks_modules_8_modules_mlp_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_mlp_modules_norm_parameters_bias_,
            1e-05,
        )
        x_135 = (
            l_self_modules_blocks_modules_8_modules_mlp_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_8_modules_mlp_modules_norm_parameters_bias_
        ) = None
        x_137 = torch._C._nn.linear(
            x_136,
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_136 = (
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_138 = torch.nn.functional.dropout(x_137, 0.0, False, False)
        x_137 = None
        x_139 = x_131 + x_138
        x_131 = x_138 = None
        x_140 = torch.nn.functional.layer_norm(
            x_139,
            (1024,),
            l_self_modules_blocks_modules_9_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_9_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_9_modules_norm1_parameters_bias_
        ) = None
        linear_36 = torch._C._nn.linear(
            x_140,
            l_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_bias_,
        )
        x_140 = (
            l_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_18 = linear_36.reshape(1, 197, 3, 16, 64)
        linear_36 = None
        qkv_9 = reshape_18.permute(2, 0, 3, 1, 4)
        reshape_18 = None
        unbind_9 = qkv_9.unbind(0)
        qkv_9 = None
        q_9 = unbind_9[0]
        k_9 = unbind_9[1]
        v_9 = unbind_9[2]
        unbind_9 = None
        x_141 = torch._C._nn.scaled_dot_product_attention(
            q_9, k_9, v_9, attn_mask=None, dropout_p=0.0
        )
        q_9 = k_9 = v_9 = None
        transpose_10 = x_141.transpose(1, 2)
        x_141 = None
        x_142 = transpose_10.reshape(1, 197, 1024)
        transpose_10 = None
        x_143 = torch.nn.functional.layer_norm(
            x_142,
            (1024,),
            l_self_modules_blocks_modules_9_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_142 = (
            l_self_modules_blocks_modules_9_modules_attn_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_9_modules_attn_modules_norm_parameters_bias_
        ) = None
        x_144 = torch._C._nn.linear(
            x_143,
            l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_,
        )
        x_143 = (
            l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_145 = torch.nn.functional.dropout(x_144, 0.0, False, False)
        x_144 = None
        x_146 = x_139 + x_145
        x_139 = x_145 = None
        x_147 = torch.nn.functional.layer_norm(
            x_146,
            (1024,),
            l_self_modules_blocks_modules_9_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_9_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_9_modules_norm2_parameters_bias_
        ) = None
        x_148 = torch._C._nn.linear(
            x_147,
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_147 = (
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_149 = torch._C._nn.gelu(x_148, approximate="none")
        x_148 = None
        x_150 = torch.nn.functional.dropout(x_149, 0.0, False, False)
        x_149 = None
        x_151 = torch.nn.functional.layer_norm(
            x_150,
            (4096,),
            l_self_modules_blocks_modules_9_modules_mlp_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_mlp_modules_norm_parameters_bias_,
            1e-05,
        )
        x_150 = (
            l_self_modules_blocks_modules_9_modules_mlp_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_9_modules_mlp_modules_norm_parameters_bias_
        ) = None
        x_152 = torch._C._nn.linear(
            x_151,
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_151 = (
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_153 = torch.nn.functional.dropout(x_152, 0.0, False, False)
        x_152 = None
        x_154 = x_146 + x_153
        x_146 = x_153 = None
        x_155 = torch.nn.functional.layer_norm(
            x_154,
            (1024,),
            l_self_modules_blocks_modules_10_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_10_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_10_modules_norm1_parameters_bias_
        ) = None
        linear_40 = torch._C._nn.linear(
            x_155,
            l_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_bias_,
        )
        x_155 = (
            l_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_20 = linear_40.reshape(1, 197, 3, 16, 64)
        linear_40 = None
        qkv_10 = reshape_20.permute(2, 0, 3, 1, 4)
        reshape_20 = None
        unbind_10 = qkv_10.unbind(0)
        qkv_10 = None
        q_10 = unbind_10[0]
        k_10 = unbind_10[1]
        v_10 = unbind_10[2]
        unbind_10 = None
        x_156 = torch._C._nn.scaled_dot_product_attention(
            q_10, k_10, v_10, attn_mask=None, dropout_p=0.0
        )
        q_10 = k_10 = v_10 = None
        transpose_11 = x_156.transpose(1, 2)
        x_156 = None
        x_157 = transpose_11.reshape(1, 197, 1024)
        transpose_11 = None
        x_158 = torch.nn.functional.layer_norm(
            x_157,
            (1024,),
            l_self_modules_blocks_modules_10_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_157 = l_self_modules_blocks_modules_10_modules_attn_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_10_modules_attn_modules_norm_parameters_bias_
        ) = None
        x_159 = torch._C._nn.linear(
            x_158,
            l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_,
        )
        x_158 = l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_160 = torch.nn.functional.dropout(x_159, 0.0, False, False)
        x_159 = None
        x_161 = x_154 + x_160
        x_154 = x_160 = None
        x_162 = torch.nn.functional.layer_norm(
            x_161,
            (1024,),
            l_self_modules_blocks_modules_10_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_10_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_10_modules_norm2_parameters_bias_
        ) = None
        x_163 = torch._C._nn.linear(
            x_162,
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_162 = (
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_164 = torch._C._nn.gelu(x_163, approximate="none")
        x_163 = None
        x_165 = torch.nn.functional.dropout(x_164, 0.0, False, False)
        x_164 = None
        x_166 = torch.nn.functional.layer_norm(
            x_165,
            (4096,),
            l_self_modules_blocks_modules_10_modules_mlp_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_mlp_modules_norm_parameters_bias_,
            1e-05,
        )
        x_165 = (
            l_self_modules_blocks_modules_10_modules_mlp_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_10_modules_mlp_modules_norm_parameters_bias_
        ) = None
        x_167 = torch._C._nn.linear(
            x_166,
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_166 = (
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_168 = torch.nn.functional.dropout(x_167, 0.0, False, False)
        x_167 = None
        x_169 = x_161 + x_168
        x_161 = x_168 = None
        x_170 = torch.nn.functional.layer_norm(
            x_169,
            (1024,),
            l_self_modules_blocks_modules_11_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_11_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_11_modules_norm1_parameters_bias_
        ) = None
        linear_44 = torch._C._nn.linear(
            x_170,
            l_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_bias_,
        )
        x_170 = (
            l_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_22 = linear_44.reshape(1, 197, 3, 16, 64)
        linear_44 = None
        qkv_11 = reshape_22.permute(2, 0, 3, 1, 4)
        reshape_22 = None
        unbind_11 = qkv_11.unbind(0)
        qkv_11 = None
        q_11 = unbind_11[0]
        k_11 = unbind_11[1]
        v_11 = unbind_11[2]
        unbind_11 = None
        x_171 = torch._C._nn.scaled_dot_product_attention(
            q_11, k_11, v_11, attn_mask=None, dropout_p=0.0
        )
        q_11 = k_11 = v_11 = None
        transpose_12 = x_171.transpose(1, 2)
        x_171 = None
        x_172 = transpose_12.reshape(1, 197, 1024)
        transpose_12 = None
        x_173 = torch.nn.functional.layer_norm(
            x_172,
            (1024,),
            l_self_modules_blocks_modules_11_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_172 = l_self_modules_blocks_modules_11_modules_attn_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_11_modules_attn_modules_norm_parameters_bias_
        ) = None
        x_174 = torch._C._nn.linear(
            x_173,
            l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_,
        )
        x_173 = l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_175 = torch.nn.functional.dropout(x_174, 0.0, False, False)
        x_174 = None
        x_176 = x_169 + x_175
        x_169 = x_175 = None
        x_177 = torch.nn.functional.layer_norm(
            x_176,
            (1024,),
            l_self_modules_blocks_modules_11_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_11_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_11_modules_norm2_parameters_bias_
        ) = None
        x_178 = torch._C._nn.linear(
            x_177,
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_177 = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_179 = torch._C._nn.gelu(x_178, approximate="none")
        x_178 = None
        x_180 = torch.nn.functional.dropout(x_179, 0.0, False, False)
        x_179 = None
        x_181 = torch.nn.functional.layer_norm(
            x_180,
            (4096,),
            l_self_modules_blocks_modules_11_modules_mlp_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_mlp_modules_norm_parameters_bias_,
            1e-05,
        )
        x_180 = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_norm_parameters_bias_
        ) = None
        x_182 = torch._C._nn.linear(
            x_181,
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_181 = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_183 = torch.nn.functional.dropout(x_182, 0.0, False, False)
        x_182 = None
        x_184 = x_176 + x_183
        x_176 = x_183 = None
        x_185 = torch.nn.functional.layer_norm(
            x_184,
            (1024,),
            l_self_modules_blocks_modules_12_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_12_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_12_modules_norm1_parameters_bias_
        ) = None
        linear_48 = torch._C._nn.linear(
            x_185,
            l_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_bias_,
        )
        x_185 = (
            l_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_24 = linear_48.reshape(1, 197, 3, 16, 64)
        linear_48 = None
        qkv_12 = reshape_24.permute(2, 0, 3, 1, 4)
        reshape_24 = None
        unbind_12 = qkv_12.unbind(0)
        qkv_12 = None
        q_12 = unbind_12[0]
        k_12 = unbind_12[1]
        v_12 = unbind_12[2]
        unbind_12 = None
        x_186 = torch._C._nn.scaled_dot_product_attention(
            q_12, k_12, v_12, attn_mask=None, dropout_p=0.0
        )
        q_12 = k_12 = v_12 = None
        transpose_13 = x_186.transpose(1, 2)
        x_186 = None
        x_187 = transpose_13.reshape(1, 197, 1024)
        transpose_13 = None
        x_188 = torch.nn.functional.layer_norm(
            x_187,
            (1024,),
            l_self_modules_blocks_modules_12_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_187 = l_self_modules_blocks_modules_12_modules_attn_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_12_modules_attn_modules_norm_parameters_bias_
        ) = None
        x_189 = torch._C._nn.linear(
            x_188,
            l_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_,
        )
        x_188 = l_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_190 = torch.nn.functional.dropout(x_189, 0.0, False, False)
        x_189 = None
        x_191 = x_184 + x_190
        x_184 = x_190 = None
        x_192 = torch.nn.functional.layer_norm(
            x_191,
            (1024,),
            l_self_modules_blocks_modules_12_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_12_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_12_modules_norm2_parameters_bias_
        ) = None
        x_193 = torch._C._nn.linear(
            x_192,
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_192 = (
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_194 = torch._C._nn.gelu(x_193, approximate="none")
        x_193 = None
        x_195 = torch.nn.functional.dropout(x_194, 0.0, False, False)
        x_194 = None
        x_196 = torch.nn.functional.layer_norm(
            x_195,
            (4096,),
            l_self_modules_blocks_modules_12_modules_mlp_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_mlp_modules_norm_parameters_bias_,
            1e-05,
        )
        x_195 = (
            l_self_modules_blocks_modules_12_modules_mlp_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_12_modules_mlp_modules_norm_parameters_bias_
        ) = None
        x_197 = torch._C._nn.linear(
            x_196,
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_196 = (
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_198 = torch.nn.functional.dropout(x_197, 0.0, False, False)
        x_197 = None
        x_199 = x_191 + x_198
        x_191 = x_198 = None
        x_200 = torch.nn.functional.layer_norm(
            x_199,
            (1024,),
            l_self_modules_blocks_modules_13_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_13_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_13_modules_norm1_parameters_bias_
        ) = None
        linear_52 = torch._C._nn.linear(
            x_200,
            l_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_bias_,
        )
        x_200 = (
            l_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_26 = linear_52.reshape(1, 197, 3, 16, 64)
        linear_52 = None
        qkv_13 = reshape_26.permute(2, 0, 3, 1, 4)
        reshape_26 = None
        unbind_13 = qkv_13.unbind(0)
        qkv_13 = None
        q_13 = unbind_13[0]
        k_13 = unbind_13[1]
        v_13 = unbind_13[2]
        unbind_13 = None
        x_201 = torch._C._nn.scaled_dot_product_attention(
            q_13, k_13, v_13, attn_mask=None, dropout_p=0.0
        )
        q_13 = k_13 = v_13 = None
        transpose_14 = x_201.transpose(1, 2)
        x_201 = None
        x_202 = transpose_14.reshape(1, 197, 1024)
        transpose_14 = None
        x_203 = torch.nn.functional.layer_norm(
            x_202,
            (1024,),
            l_self_modules_blocks_modules_13_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_202 = l_self_modules_blocks_modules_13_modules_attn_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_13_modules_attn_modules_norm_parameters_bias_
        ) = None
        x_204 = torch._C._nn.linear(
            x_203,
            l_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_,
        )
        x_203 = l_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_205 = torch.nn.functional.dropout(x_204, 0.0, False, False)
        x_204 = None
        x_206 = x_199 + x_205
        x_199 = x_205 = None
        x_207 = torch.nn.functional.layer_norm(
            x_206,
            (1024,),
            l_self_modules_blocks_modules_13_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_13_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_13_modules_norm2_parameters_bias_
        ) = None
        x_208 = torch._C._nn.linear(
            x_207,
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_207 = (
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_209 = torch._C._nn.gelu(x_208, approximate="none")
        x_208 = None
        x_210 = torch.nn.functional.dropout(x_209, 0.0, False, False)
        x_209 = None
        x_211 = torch.nn.functional.layer_norm(
            x_210,
            (4096,),
            l_self_modules_blocks_modules_13_modules_mlp_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_mlp_modules_norm_parameters_bias_,
            1e-05,
        )
        x_210 = (
            l_self_modules_blocks_modules_13_modules_mlp_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_13_modules_mlp_modules_norm_parameters_bias_
        ) = None
        x_212 = torch._C._nn.linear(
            x_211,
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_211 = (
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_213 = torch.nn.functional.dropout(x_212, 0.0, False, False)
        x_212 = None
        x_214 = x_206 + x_213
        x_206 = x_213 = None
        x_215 = torch.nn.functional.layer_norm(
            x_214,
            (1024,),
            l_self_modules_blocks_modules_14_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_14_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_14_modules_norm1_parameters_bias_
        ) = None
        linear_56 = torch._C._nn.linear(
            x_215,
            l_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_bias_,
        )
        x_215 = (
            l_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_28 = linear_56.reshape(1, 197, 3, 16, 64)
        linear_56 = None
        qkv_14 = reshape_28.permute(2, 0, 3, 1, 4)
        reshape_28 = None
        unbind_14 = qkv_14.unbind(0)
        qkv_14 = None
        q_14 = unbind_14[0]
        k_14 = unbind_14[1]
        v_14 = unbind_14[2]
        unbind_14 = None
        x_216 = torch._C._nn.scaled_dot_product_attention(
            q_14, k_14, v_14, attn_mask=None, dropout_p=0.0
        )
        q_14 = k_14 = v_14 = None
        transpose_15 = x_216.transpose(1, 2)
        x_216 = None
        x_217 = transpose_15.reshape(1, 197, 1024)
        transpose_15 = None
        x_218 = torch.nn.functional.layer_norm(
            x_217,
            (1024,),
            l_self_modules_blocks_modules_14_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_217 = l_self_modules_blocks_modules_14_modules_attn_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_14_modules_attn_modules_norm_parameters_bias_
        ) = None
        x_219 = torch._C._nn.linear(
            x_218,
            l_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_bias_,
        )
        x_218 = l_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_220 = torch.nn.functional.dropout(x_219, 0.0, False, False)
        x_219 = None
        x_221 = x_214 + x_220
        x_214 = x_220 = None
        x_222 = torch.nn.functional.layer_norm(
            x_221,
            (1024,),
            l_self_modules_blocks_modules_14_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_14_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_14_modules_norm2_parameters_bias_
        ) = None
        x_223 = torch._C._nn.linear(
            x_222,
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_222 = (
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_224 = torch._C._nn.gelu(x_223, approximate="none")
        x_223 = None
        x_225 = torch.nn.functional.dropout(x_224, 0.0, False, False)
        x_224 = None
        x_226 = torch.nn.functional.layer_norm(
            x_225,
            (4096,),
            l_self_modules_blocks_modules_14_modules_mlp_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_mlp_modules_norm_parameters_bias_,
            1e-05,
        )
        x_225 = (
            l_self_modules_blocks_modules_14_modules_mlp_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_14_modules_mlp_modules_norm_parameters_bias_
        ) = None
        x_227 = torch._C._nn.linear(
            x_226,
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_226 = (
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_228 = torch.nn.functional.dropout(x_227, 0.0, False, False)
        x_227 = None
        x_229 = x_221 + x_228
        x_221 = x_228 = None
        x_230 = torch.nn.functional.layer_norm(
            x_229,
            (1024,),
            l_self_modules_blocks_modules_15_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_15_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_15_modules_norm1_parameters_bias_
        ) = None
        linear_60 = torch._C._nn.linear(
            x_230,
            l_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_bias_,
        )
        x_230 = (
            l_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_30 = linear_60.reshape(1, 197, 3, 16, 64)
        linear_60 = None
        qkv_15 = reshape_30.permute(2, 0, 3, 1, 4)
        reshape_30 = None
        unbind_15 = qkv_15.unbind(0)
        qkv_15 = None
        q_15 = unbind_15[0]
        k_15 = unbind_15[1]
        v_15 = unbind_15[2]
        unbind_15 = None
        x_231 = torch._C._nn.scaled_dot_product_attention(
            q_15, k_15, v_15, attn_mask=None, dropout_p=0.0
        )
        q_15 = k_15 = v_15 = None
        transpose_16 = x_231.transpose(1, 2)
        x_231 = None
        x_232 = transpose_16.reshape(1, 197, 1024)
        transpose_16 = None
        x_233 = torch.nn.functional.layer_norm(
            x_232,
            (1024,),
            l_self_modules_blocks_modules_15_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_232 = l_self_modules_blocks_modules_15_modules_attn_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_15_modules_attn_modules_norm_parameters_bias_
        ) = None
        x_234 = torch._C._nn.linear(
            x_233,
            l_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_bias_,
        )
        x_233 = l_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_235 = torch.nn.functional.dropout(x_234, 0.0, False, False)
        x_234 = None
        x_236 = x_229 + x_235
        x_229 = x_235 = None
        x_237 = torch.nn.functional.layer_norm(
            x_236,
            (1024,),
            l_self_modules_blocks_modules_15_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_15_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_15_modules_norm2_parameters_bias_
        ) = None
        x_238 = torch._C._nn.linear(
            x_237,
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_237 = (
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_239 = torch._C._nn.gelu(x_238, approximate="none")
        x_238 = None
        x_240 = torch.nn.functional.dropout(x_239, 0.0, False, False)
        x_239 = None
        x_241 = torch.nn.functional.layer_norm(
            x_240,
            (4096,),
            l_self_modules_blocks_modules_15_modules_mlp_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_mlp_modules_norm_parameters_bias_,
            1e-05,
        )
        x_240 = (
            l_self_modules_blocks_modules_15_modules_mlp_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_15_modules_mlp_modules_norm_parameters_bias_
        ) = None
        x_242 = torch._C._nn.linear(
            x_241,
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_241 = (
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_243 = torch.nn.functional.dropout(x_242, 0.0, False, False)
        x_242 = None
        x_244 = x_236 + x_243
        x_236 = x_243 = None
        x_245 = torch.nn.functional.layer_norm(
            x_244,
            (1024,),
            l_self_modules_blocks_modules_16_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_16_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_16_modules_norm1_parameters_bias_
        ) = None
        linear_64 = torch._C._nn.linear(
            x_245,
            l_self_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_bias_,
        )
        x_245 = (
            l_self_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_32 = linear_64.reshape(1, 197, 3, 16, 64)
        linear_64 = None
        qkv_16 = reshape_32.permute(2, 0, 3, 1, 4)
        reshape_32 = None
        unbind_16 = qkv_16.unbind(0)
        qkv_16 = None
        q_16 = unbind_16[0]
        k_16 = unbind_16[1]
        v_16 = unbind_16[2]
        unbind_16 = None
        x_246 = torch._C._nn.scaled_dot_product_attention(
            q_16, k_16, v_16, attn_mask=None, dropout_p=0.0
        )
        q_16 = k_16 = v_16 = None
        transpose_17 = x_246.transpose(1, 2)
        x_246 = None
        x_247 = transpose_17.reshape(1, 197, 1024)
        transpose_17 = None
        x_248 = torch.nn.functional.layer_norm(
            x_247,
            (1024,),
            l_self_modules_blocks_modules_16_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_247 = l_self_modules_blocks_modules_16_modules_attn_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_16_modules_attn_modules_norm_parameters_bias_
        ) = None
        x_249 = torch._C._nn.linear(
            x_248,
            l_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_bias_,
        )
        x_248 = l_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_250 = torch.nn.functional.dropout(x_249, 0.0, False, False)
        x_249 = None
        x_251 = x_244 + x_250
        x_244 = x_250 = None
        x_252 = torch.nn.functional.layer_norm(
            x_251,
            (1024,),
            l_self_modules_blocks_modules_16_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_16_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_16_modules_norm2_parameters_bias_
        ) = None
        x_253 = torch._C._nn.linear(
            x_252,
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_252 = (
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_254 = torch._C._nn.gelu(x_253, approximate="none")
        x_253 = None
        x_255 = torch.nn.functional.dropout(x_254, 0.0, False, False)
        x_254 = None
        x_256 = torch.nn.functional.layer_norm(
            x_255,
            (4096,),
            l_self_modules_blocks_modules_16_modules_mlp_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_mlp_modules_norm_parameters_bias_,
            1e-05,
        )
        x_255 = (
            l_self_modules_blocks_modules_16_modules_mlp_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_16_modules_mlp_modules_norm_parameters_bias_
        ) = None
        x_257 = torch._C._nn.linear(
            x_256,
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_256 = (
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_258 = torch.nn.functional.dropout(x_257, 0.0, False, False)
        x_257 = None
        x_259 = x_251 + x_258
        x_251 = x_258 = None
        x_260 = torch.nn.functional.layer_norm(
            x_259,
            (1024,),
            l_self_modules_blocks_modules_17_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_17_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_17_modules_norm1_parameters_bias_
        ) = None
        linear_68 = torch._C._nn.linear(
            x_260,
            l_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_bias_,
        )
        x_260 = (
            l_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_34 = linear_68.reshape(1, 197, 3, 16, 64)
        linear_68 = None
        qkv_17 = reshape_34.permute(2, 0, 3, 1, 4)
        reshape_34 = None
        unbind_17 = qkv_17.unbind(0)
        qkv_17 = None
        q_17 = unbind_17[0]
        k_17 = unbind_17[1]
        v_17 = unbind_17[2]
        unbind_17 = None
        x_261 = torch._C._nn.scaled_dot_product_attention(
            q_17, k_17, v_17, attn_mask=None, dropout_p=0.0
        )
        q_17 = k_17 = v_17 = None
        transpose_18 = x_261.transpose(1, 2)
        x_261 = None
        x_262 = transpose_18.reshape(1, 197, 1024)
        transpose_18 = None
        x_263 = torch.nn.functional.layer_norm(
            x_262,
            (1024,),
            l_self_modules_blocks_modules_17_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_262 = l_self_modules_blocks_modules_17_modules_attn_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_17_modules_attn_modules_norm_parameters_bias_
        ) = None
        x_264 = torch._C._nn.linear(
            x_263,
            l_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_bias_,
        )
        x_263 = l_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_265 = torch.nn.functional.dropout(x_264, 0.0, False, False)
        x_264 = None
        x_266 = x_259 + x_265
        x_259 = x_265 = None
        x_267 = torch.nn.functional.layer_norm(
            x_266,
            (1024,),
            l_self_modules_blocks_modules_17_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_17_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_17_modules_norm2_parameters_bias_
        ) = None
        x_268 = torch._C._nn.linear(
            x_267,
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_267 = (
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_269 = torch._C._nn.gelu(x_268, approximate="none")
        x_268 = None
        x_270 = torch.nn.functional.dropout(x_269, 0.0, False, False)
        x_269 = None
        x_271 = torch.nn.functional.layer_norm(
            x_270,
            (4096,),
            l_self_modules_blocks_modules_17_modules_mlp_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_mlp_modules_norm_parameters_bias_,
            1e-05,
        )
        x_270 = (
            l_self_modules_blocks_modules_17_modules_mlp_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_17_modules_mlp_modules_norm_parameters_bias_
        ) = None
        x_272 = torch._C._nn.linear(
            x_271,
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_271 = (
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_273 = torch.nn.functional.dropout(x_272, 0.0, False, False)
        x_272 = None
        x_274 = x_266 + x_273
        x_266 = x_273 = None
        x_275 = torch.nn.functional.layer_norm(
            x_274,
            (1024,),
            l_self_modules_blocks_modules_18_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_18_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_18_modules_norm1_parameters_bias_
        ) = None
        linear_72 = torch._C._nn.linear(
            x_275,
            l_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_bias_,
        )
        x_275 = (
            l_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_36 = linear_72.reshape(1, 197, 3, 16, 64)
        linear_72 = None
        qkv_18 = reshape_36.permute(2, 0, 3, 1, 4)
        reshape_36 = None
        unbind_18 = qkv_18.unbind(0)
        qkv_18 = None
        q_18 = unbind_18[0]
        k_18 = unbind_18[1]
        v_18 = unbind_18[2]
        unbind_18 = None
        x_276 = torch._C._nn.scaled_dot_product_attention(
            q_18, k_18, v_18, attn_mask=None, dropout_p=0.0
        )
        q_18 = k_18 = v_18 = None
        transpose_19 = x_276.transpose(1, 2)
        x_276 = None
        x_277 = transpose_19.reshape(1, 197, 1024)
        transpose_19 = None
        x_278 = torch.nn.functional.layer_norm(
            x_277,
            (1024,),
            l_self_modules_blocks_modules_18_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_277 = l_self_modules_blocks_modules_18_modules_attn_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_18_modules_attn_modules_norm_parameters_bias_
        ) = None
        x_279 = torch._C._nn.linear(
            x_278,
            l_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_bias_,
        )
        x_278 = l_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_280 = torch.nn.functional.dropout(x_279, 0.0, False, False)
        x_279 = None
        x_281 = x_274 + x_280
        x_274 = x_280 = None
        x_282 = torch.nn.functional.layer_norm(
            x_281,
            (1024,),
            l_self_modules_blocks_modules_18_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_18_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_18_modules_norm2_parameters_bias_
        ) = None
        x_283 = torch._C._nn.linear(
            x_282,
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_282 = (
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_284 = torch._C._nn.gelu(x_283, approximate="none")
        x_283 = None
        x_285 = torch.nn.functional.dropout(x_284, 0.0, False, False)
        x_284 = None
        x_286 = torch.nn.functional.layer_norm(
            x_285,
            (4096,),
            l_self_modules_blocks_modules_18_modules_mlp_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_mlp_modules_norm_parameters_bias_,
            1e-05,
        )
        x_285 = (
            l_self_modules_blocks_modules_18_modules_mlp_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_18_modules_mlp_modules_norm_parameters_bias_
        ) = None
        x_287 = torch._C._nn.linear(
            x_286,
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_286 = (
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_288 = torch.nn.functional.dropout(x_287, 0.0, False, False)
        x_287 = None
        x_289 = x_281 + x_288
        x_281 = x_288 = None
        x_290 = torch.nn.functional.layer_norm(
            x_289,
            (1024,),
            l_self_modules_blocks_modules_19_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_19_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_19_modules_norm1_parameters_bias_
        ) = None
        linear_76 = torch._C._nn.linear(
            x_290,
            l_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_bias_,
        )
        x_290 = (
            l_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_38 = linear_76.reshape(1, 197, 3, 16, 64)
        linear_76 = None
        qkv_19 = reshape_38.permute(2, 0, 3, 1, 4)
        reshape_38 = None
        unbind_19 = qkv_19.unbind(0)
        qkv_19 = None
        q_19 = unbind_19[0]
        k_19 = unbind_19[1]
        v_19 = unbind_19[2]
        unbind_19 = None
        x_291 = torch._C._nn.scaled_dot_product_attention(
            q_19, k_19, v_19, attn_mask=None, dropout_p=0.0
        )
        q_19 = k_19 = v_19 = None
        transpose_20 = x_291.transpose(1, 2)
        x_291 = None
        x_292 = transpose_20.reshape(1, 197, 1024)
        transpose_20 = None
        x_293 = torch.nn.functional.layer_norm(
            x_292,
            (1024,),
            l_self_modules_blocks_modules_19_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_292 = l_self_modules_blocks_modules_19_modules_attn_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_19_modules_attn_modules_norm_parameters_bias_
        ) = None
        x_294 = torch._C._nn.linear(
            x_293,
            l_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_bias_,
        )
        x_293 = l_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_295 = torch.nn.functional.dropout(x_294, 0.0, False, False)
        x_294 = None
        x_296 = x_289 + x_295
        x_289 = x_295 = None
        x_297 = torch.nn.functional.layer_norm(
            x_296,
            (1024,),
            l_self_modules_blocks_modules_19_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_19_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_19_modules_norm2_parameters_bias_
        ) = None
        x_298 = torch._C._nn.linear(
            x_297,
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_297 = (
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_299 = torch._C._nn.gelu(x_298, approximate="none")
        x_298 = None
        x_300 = torch.nn.functional.dropout(x_299, 0.0, False, False)
        x_299 = None
        x_301 = torch.nn.functional.layer_norm(
            x_300,
            (4096,),
            l_self_modules_blocks_modules_19_modules_mlp_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_mlp_modules_norm_parameters_bias_,
            1e-05,
        )
        x_300 = (
            l_self_modules_blocks_modules_19_modules_mlp_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_19_modules_mlp_modules_norm_parameters_bias_
        ) = None
        x_302 = torch._C._nn.linear(
            x_301,
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_301 = (
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_303 = torch.nn.functional.dropout(x_302, 0.0, False, False)
        x_302 = None
        x_304 = x_296 + x_303
        x_296 = x_303 = None
        x_305 = torch.nn.functional.layer_norm(
            x_304,
            (1024,),
            l_self_modules_blocks_modules_20_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_20_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_20_modules_norm1_parameters_bias_
        ) = None
        linear_80 = torch._C._nn.linear(
            x_305,
            l_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_bias_,
        )
        x_305 = (
            l_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_40 = linear_80.reshape(1, 197, 3, 16, 64)
        linear_80 = None
        qkv_20 = reshape_40.permute(2, 0, 3, 1, 4)
        reshape_40 = None
        unbind_20 = qkv_20.unbind(0)
        qkv_20 = None
        q_20 = unbind_20[0]
        k_20 = unbind_20[1]
        v_20 = unbind_20[2]
        unbind_20 = None
        x_306 = torch._C._nn.scaled_dot_product_attention(
            q_20, k_20, v_20, attn_mask=None, dropout_p=0.0
        )
        q_20 = k_20 = v_20 = None
        transpose_21 = x_306.transpose(1, 2)
        x_306 = None
        x_307 = transpose_21.reshape(1, 197, 1024)
        transpose_21 = None
        x_308 = torch.nn.functional.layer_norm(
            x_307,
            (1024,),
            l_self_modules_blocks_modules_20_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_307 = l_self_modules_blocks_modules_20_modules_attn_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_20_modules_attn_modules_norm_parameters_bias_
        ) = None
        x_309 = torch._C._nn.linear(
            x_308,
            l_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_bias_,
        )
        x_308 = l_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_310 = torch.nn.functional.dropout(x_309, 0.0, False, False)
        x_309 = None
        x_311 = x_304 + x_310
        x_304 = x_310 = None
        x_312 = torch.nn.functional.layer_norm(
            x_311,
            (1024,),
            l_self_modules_blocks_modules_20_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_20_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_20_modules_norm2_parameters_bias_
        ) = None
        x_313 = torch._C._nn.linear(
            x_312,
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_312 = (
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_314 = torch._C._nn.gelu(x_313, approximate="none")
        x_313 = None
        x_315 = torch.nn.functional.dropout(x_314, 0.0, False, False)
        x_314 = None
        x_316 = torch.nn.functional.layer_norm(
            x_315,
            (4096,),
            l_self_modules_blocks_modules_20_modules_mlp_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_mlp_modules_norm_parameters_bias_,
            1e-05,
        )
        x_315 = (
            l_self_modules_blocks_modules_20_modules_mlp_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_20_modules_mlp_modules_norm_parameters_bias_
        ) = None
        x_317 = torch._C._nn.linear(
            x_316,
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_316 = (
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_318 = torch.nn.functional.dropout(x_317, 0.0, False, False)
        x_317 = None
        x_319 = x_311 + x_318
        x_311 = x_318 = None
        x_320 = torch.nn.functional.layer_norm(
            x_319,
            (1024,),
            l_self_modules_blocks_modules_21_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_21_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_21_modules_norm1_parameters_bias_
        ) = None
        linear_84 = torch._C._nn.linear(
            x_320,
            l_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_bias_,
        )
        x_320 = (
            l_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_42 = linear_84.reshape(1, 197, 3, 16, 64)
        linear_84 = None
        qkv_21 = reshape_42.permute(2, 0, 3, 1, 4)
        reshape_42 = None
        unbind_21 = qkv_21.unbind(0)
        qkv_21 = None
        q_21 = unbind_21[0]
        k_21 = unbind_21[1]
        v_21 = unbind_21[2]
        unbind_21 = None
        x_321 = torch._C._nn.scaled_dot_product_attention(
            q_21, k_21, v_21, attn_mask=None, dropout_p=0.0
        )
        q_21 = k_21 = v_21 = None
        transpose_22 = x_321.transpose(1, 2)
        x_321 = None
        x_322 = transpose_22.reshape(1, 197, 1024)
        transpose_22 = None
        x_323 = torch.nn.functional.layer_norm(
            x_322,
            (1024,),
            l_self_modules_blocks_modules_21_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_322 = l_self_modules_blocks_modules_21_modules_attn_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_21_modules_attn_modules_norm_parameters_bias_
        ) = None
        x_324 = torch._C._nn.linear(
            x_323,
            l_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_bias_,
        )
        x_323 = l_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_325 = torch.nn.functional.dropout(x_324, 0.0, False, False)
        x_324 = None
        x_326 = x_319 + x_325
        x_319 = x_325 = None
        x_327 = torch.nn.functional.layer_norm(
            x_326,
            (1024,),
            l_self_modules_blocks_modules_21_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_21_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_21_modules_norm2_parameters_bias_
        ) = None
        x_328 = torch._C._nn.linear(
            x_327,
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_327 = (
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_329 = torch._C._nn.gelu(x_328, approximate="none")
        x_328 = None
        x_330 = torch.nn.functional.dropout(x_329, 0.0, False, False)
        x_329 = None
        x_331 = torch.nn.functional.layer_norm(
            x_330,
            (4096,),
            l_self_modules_blocks_modules_21_modules_mlp_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_mlp_modules_norm_parameters_bias_,
            1e-05,
        )
        x_330 = (
            l_self_modules_blocks_modules_21_modules_mlp_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_21_modules_mlp_modules_norm_parameters_bias_
        ) = None
        x_332 = torch._C._nn.linear(
            x_331,
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_331 = (
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_333 = torch.nn.functional.dropout(x_332, 0.0, False, False)
        x_332 = None
        x_334 = x_326 + x_333
        x_326 = x_333 = None
        x_335 = torch.nn.functional.layer_norm(
            x_334,
            (1024,),
            l_self_modules_blocks_modules_22_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_22_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_22_modules_norm1_parameters_bias_
        ) = None
        linear_88 = torch._C._nn.linear(
            x_335,
            l_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_bias_,
        )
        x_335 = (
            l_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_44 = linear_88.reshape(1, 197, 3, 16, 64)
        linear_88 = None
        qkv_22 = reshape_44.permute(2, 0, 3, 1, 4)
        reshape_44 = None
        unbind_22 = qkv_22.unbind(0)
        qkv_22 = None
        q_22 = unbind_22[0]
        k_22 = unbind_22[1]
        v_22 = unbind_22[2]
        unbind_22 = None
        x_336 = torch._C._nn.scaled_dot_product_attention(
            q_22, k_22, v_22, attn_mask=None, dropout_p=0.0
        )
        q_22 = k_22 = v_22 = None
        transpose_23 = x_336.transpose(1, 2)
        x_336 = None
        x_337 = transpose_23.reshape(1, 197, 1024)
        transpose_23 = None
        x_338 = torch.nn.functional.layer_norm(
            x_337,
            (1024,),
            l_self_modules_blocks_modules_22_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_337 = l_self_modules_blocks_modules_22_modules_attn_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_22_modules_attn_modules_norm_parameters_bias_
        ) = None
        x_339 = torch._C._nn.linear(
            x_338,
            l_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_bias_,
        )
        x_338 = l_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_340 = torch.nn.functional.dropout(x_339, 0.0, False, False)
        x_339 = None
        x_341 = x_334 + x_340
        x_334 = x_340 = None
        x_342 = torch.nn.functional.layer_norm(
            x_341,
            (1024,),
            l_self_modules_blocks_modules_22_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_22_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_22_modules_norm2_parameters_bias_
        ) = None
        x_343 = torch._C._nn.linear(
            x_342,
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_342 = (
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_344 = torch._C._nn.gelu(x_343, approximate="none")
        x_343 = None
        x_345 = torch.nn.functional.dropout(x_344, 0.0, False, False)
        x_344 = None
        x_346 = torch.nn.functional.layer_norm(
            x_345,
            (4096,),
            l_self_modules_blocks_modules_22_modules_mlp_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_mlp_modules_norm_parameters_bias_,
            1e-05,
        )
        x_345 = (
            l_self_modules_blocks_modules_22_modules_mlp_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_22_modules_mlp_modules_norm_parameters_bias_
        ) = None
        x_347 = torch._C._nn.linear(
            x_346,
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_346 = (
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_348 = torch.nn.functional.dropout(x_347, 0.0, False, False)
        x_347 = None
        x_349 = x_341 + x_348
        x_341 = x_348 = None
        x_350 = torch.nn.functional.layer_norm(
            x_349,
            (1024,),
            l_self_modules_blocks_modules_23_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_23_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_23_modules_norm1_parameters_bias_
        ) = None
        linear_92 = torch._C._nn.linear(
            x_350,
            l_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_bias_,
        )
        x_350 = (
            l_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_bias_
        ) = None
        reshape_46 = linear_92.reshape(1, 197, 3, 16, 64)
        linear_92 = None
        qkv_23 = reshape_46.permute(2, 0, 3, 1, 4)
        reshape_46 = None
        unbind_23 = qkv_23.unbind(0)
        qkv_23 = None
        q_23 = unbind_23[0]
        k_23 = unbind_23[1]
        v_23 = unbind_23[2]
        unbind_23 = None
        x_351 = torch._C._nn.scaled_dot_product_attention(
            q_23, k_23, v_23, attn_mask=None, dropout_p=0.0
        )
        q_23 = k_23 = v_23 = None
        transpose_24 = x_351.transpose(1, 2)
        x_351 = None
        x_352 = transpose_24.reshape(1, 197, 1024)
        transpose_24 = None
        x_353 = torch.nn.functional.layer_norm(
            x_352,
            (1024,),
            l_self_modules_blocks_modules_23_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_attn_modules_norm_parameters_bias_,
            1e-05,
        )
        x_352 = l_self_modules_blocks_modules_23_modules_attn_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_23_modules_attn_modules_norm_parameters_bias_
        ) = None
        x_354 = torch._C._nn.linear(
            x_353,
            l_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_bias_,
        )
        x_353 = l_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_355 = torch.nn.functional.dropout(x_354, 0.0, False, False)
        x_354 = None
        x_356 = x_349 + x_355
        x_349 = x_355 = None
        x_357 = torch.nn.functional.layer_norm(
            x_356,
            (1024,),
            l_self_modules_blocks_modules_23_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_23_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_23_modules_norm2_parameters_bias_
        ) = None
        x_358 = torch._C._nn.linear(
            x_357,
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_357 = (
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_359 = torch._C._nn.gelu(x_358, approximate="none")
        x_358 = None
        x_360 = torch.nn.functional.dropout(x_359, 0.0, False, False)
        x_359 = None
        x_361 = torch.nn.functional.layer_norm(
            x_360,
            (4096,),
            l_self_modules_blocks_modules_23_modules_mlp_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_mlp_modules_norm_parameters_bias_,
            1e-05,
        )
        x_360 = (
            l_self_modules_blocks_modules_23_modules_mlp_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_23_modules_mlp_modules_norm_parameters_bias_
        ) = None
        x_362 = torch._C._nn.linear(
            x_361,
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_361 = (
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_363 = torch.nn.functional.dropout(x_362, 0.0, False, False)
        x_362 = None
        x_364 = x_356 + x_363
        x_356 = x_363 = None
        x_365 = x_364[
            (slice(None, None, None), slice(l_self_num_prefix_tokens, None, None))
        ]
        x_364 = l_self_num_prefix_tokens = None
        x_366 = x_365.mean(dim=1)
        x_365 = None
        x_367 = torch.nn.functional.layer_norm(
            x_366,
            (1024,),
            l_self_modules_fc_norm_parameters_weight_,
            l_self_modules_fc_norm_parameters_bias_,
            1e-05,
        )
        x_366 = (
            l_self_modules_fc_norm_parameters_weight_
        ) = l_self_modules_fc_norm_parameters_bias_ = None
        x_368 = torch.nn.functional.dropout(x_367, 0.0, False, False)
        x_367 = None
        x_369 = torch._C._nn.linear(
            x_368,
            l_self_modules_head_parameters_weight_,
            l_self_modules_head_parameters_bias_,
        )
        x_368 = (
            l_self_modules_head_parameters_weight_
        ) = l_self_modules_head_parameters_bias_ = None
        return (x_369,)
