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
        reshape = posemb.reshape(1, 64, 64, 768)
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
            (768,),
            l_self_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_0_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_0_modules_norm1_parameters_bias_
        ) = None
        x_5 = torch._C._nn.pad(x_4, (0, 0, 0, 0, 0, 0), "constant", None)
        x_4 = None
        x_6 = x_5.view(1, 1, 14, 1, 14, 768)
        x_5 = None
        permute_3 = x_6.permute(0, 1, 3, 2, 4, 5)
        x_6 = None
        contiguous = permute_3.contiguous()
        permute_3 = None
        windows = contiguous.view(-1, 14, 14, 768)
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
        view_2 = linear.view(1, 196, 3, 12, -1)
        linear = None
        qkv = view_2.permute(2, 0, 3, 1, 4)
        view_2 = None
        reshape_2 = qkv.reshape(3, 12, 196, -1)
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
        r_q = q.reshape(12, 14, 14, 64)
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
        view_3 = x_8.view(1, 12, 196, -1)
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
            (768,),
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
            (768,),
            l_self_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_1_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_1_modules_norm1_parameters_bias_
        ) = None
        x_26 = torch._C._nn.pad(x_25, (0, 0, 0, 0, 0, 0), "constant", None)
        x_25 = None
        x_27 = x_26.view(1, 1, 14, 1, 14, 768)
        x_26 = None
        permute_6 = x_27.permute(0, 1, 3, 2, 4, 5)
        x_27 = None
        contiguous_3 = permute_6.contiguous()
        permute_6 = None
        windows_1 = contiguous_3.view(-1, 14, 14, 768)
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
        view_9 = linear_4.view(1, 196, 3, 12, -1)
        linear_4 = None
        qkv_1 = view_9.permute(2, 0, 3, 1, 4)
        view_9 = None
        reshape_9 = qkv_1.reshape(3, 12, 196, -1)
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
        r_q_1 = q_1.reshape(12, 14, 14, 64)
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
        view_10 = x_29.view(1, 12, 196, -1)
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
            (768,),
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
            (768,),
            l_self_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_2_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_2_modules_norm1_parameters_bias_
        ) = None
        x_47 = x_46.reshape(1, 196, -1)
        x_46 = None
        linear_8 = torch._C._nn.linear(
            x_47,
            l_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_,
        )
        x_47 = (
            l_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_14 = linear_8.view(1, 196, 3, 12, -1)
        linear_8 = None
        qkv_2 = view_14.permute(2, 0, 3, 1, 4)
        view_14 = None
        reshape_16 = qkv_2.reshape(3, 12, 196, -1)
        qkv_2 = None
        unbind_2 = reshape_16.unbind(0)
        reshape_16 = None
        q_2 = unbind_2[0]
        k_2 = unbind_2[1]
        v_2 = unbind_2[2]
        unbind_2 = None
        reshape_17 = (
            l_self_modules_blocks_modules_2_modules_attn_parameters_rel_pos_h_.reshape(
                1, 127, -1
            )
        )
        l_self_modules_blocks_modules_2_modules_attn_parameters_rel_pos_h_ = None
        permute_10 = reshape_17.permute(0, 2, 1)
        reshape_17 = None
        rel_pos_resized = torch.nn.functional.interpolate(
            permute_10, size=27, mode="linear"
        )
        permute_10 = None
        reshape_18 = rel_pos_resized.reshape(-1, 27)
        rel_pos_resized = None
        rel_pos_resized_1 = reshape_18.permute(1, 0)
        reshape_18 = None
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
        Rh_2 = rel_pos_resized_1[long_4]
        rel_pos_resized_1 = long_4 = None
        reshape_19 = (
            l_self_modules_blocks_modules_2_modules_attn_parameters_rel_pos_w_.reshape(
                1, 127, -1
            )
        )
        l_self_modules_blocks_modules_2_modules_attn_parameters_rel_pos_w_ = None
        permute_12 = reshape_19.permute(0, 2, 1)
        reshape_19 = None
        rel_pos_resized_2 = torch.nn.functional.interpolate(
            permute_12, size=27, mode="linear"
        )
        permute_12 = None
        reshape_20 = rel_pos_resized_2.reshape(-1, 27)
        rel_pos_resized_2 = None
        rel_pos_resized_3 = reshape_20.permute(1, 0)
        reshape_20 = None
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
        Rw_2 = rel_pos_resized_3[long_5]
        rel_pos_resized_3 = long_5 = None
        r_q_2 = q_2.reshape(12, 14, 14, 64)
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
        x_48 = torch._C._nn.scaled_dot_product_attention(
            q_2, k_2, v_2, attn_mask=attn_bias_5, dropout_p=0.0
        )
        q_2 = k_2 = v_2 = attn_bias_5 = None
        view_15 = x_48.view(1, 12, 196, -1)
        x_48 = None
        transpose_2 = view_15.transpose(1, 2)
        view_15 = None
        x_49 = transpose_2.reshape(1, 196, -1)
        transpose_2 = None
        x_50 = torch._C._nn.linear(
            x_49,
            l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_,
        )
        x_49 = (
            l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_51 = torch.nn.functional.dropout(x_50, 0.0, False, False)
        x_50 = None
        x_52 = x_51.view(1, 14, 14, -1)
        x_51 = None
        x_53 = x_45 + x_52
        x_45 = x_52 = None
        x_54 = x_53.reshape(1, 196, -1)
        x_53 = None
        layer_norm_5 = torch.nn.functional.layer_norm(
            x_54,
            (768,),
            l_self_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_2_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_2_modules_norm2_parameters_bias_
        ) = None
        x_55 = torch._C._nn.linear(
            layer_norm_5,
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_5 = (
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_56 = torch._C._nn.gelu(x_55, approximate="none")
        x_55 = None
        x_57 = torch.nn.functional.dropout(x_56, 0.0, False, False)
        x_56 = None
        x_58 = torch._C._nn.linear(
            x_57,
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_57 = (
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_59 = torch.nn.functional.dropout(x_58, 0.0, False, False)
        x_58 = None
        x_60 = x_54 + x_59
        x_54 = x_59 = None
        x_61 = x_60.reshape(1, 14, 14, -1)
        x_60 = None
        x_62 = torch.nn.functional.layer_norm(
            x_61,
            (768,),
            l_self_modules_blocks_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_3_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_3_modules_norm1_parameters_bias_
        ) = None
        x_63 = torch._C._nn.pad(x_62, (0, 0, 0, 0, 0, 0), "constant", None)
        x_62 = None
        x_64 = x_63.view(1, 1, 14, 1, 14, 768)
        x_63 = None
        permute_14 = x_64.permute(0, 1, 3, 2, 4, 5)
        x_64 = None
        contiguous_6 = permute_14.contiguous()
        permute_14 = None
        windows_2 = contiguous_6.view(-1, 14, 14, 768)
        contiguous_6 = None
        x_65 = windows_2.reshape(1, 196, -1)
        windows_2 = None
        linear_12 = torch._C._nn.linear(
            x_65,
            l_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_bias_,
        )
        x_65 = (
            l_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_19 = linear_12.view(1, 196, 3, 12, -1)
        linear_12 = None
        qkv_3 = view_19.permute(2, 0, 3, 1, 4)
        view_19 = None
        reshape_27 = qkv_3.reshape(3, 12, 196, -1)
        qkv_3 = None
        unbind_3 = reshape_27.unbind(0)
        reshape_27 = None
        q_3 = unbind_3[0]
        k_3 = unbind_3[1]
        v_3 = unbind_3[2]
        unbind_3 = None
        arange_12 = torch.arange(14)
        getitem_42 = arange_12[(slice(None, None, None), None)]
        arange_12 = None
        q_coords_6 = getitem_42 * 1.0
        getitem_42 = None
        arange_13 = torch.arange(14)
        getitem_43 = arange_13[(None, slice(None, None, None))]
        arange_13 = None
        k_coords_6 = getitem_43 * 1.0
        getitem_43 = None
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
        getitem_45 = arange_14[(slice(None, None, None), None)]
        arange_14 = None
        q_coords_7 = getitem_45 * 1.0
        getitem_45 = None
        arange_15 = torch.arange(14)
        getitem_46 = arange_15[(None, slice(None, None, None))]
        arange_15 = None
        k_coords_7 = getitem_46 * 1.0
        getitem_46 = None
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
        r_q_3 = q_3.reshape(12, 14, 14, 64)
        rel_h_3 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_3, Rh_3)
        Rh_3 = None
        rel_w_3 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_3, Rw_3)
        r_q_3 = Rw_3 = None
        getitem_48 = rel_h_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_3 = None
        getitem_49 = rel_w_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_3 = None
        attn_bias_6 = getitem_48 + getitem_49
        getitem_48 = getitem_49 = None
        attn_bias_7 = attn_bias_6.reshape(-1, 196, 196)
        attn_bias_6 = None
        x_66 = torch._C._nn.scaled_dot_product_attention(
            q_3, k_3, v_3, attn_mask=attn_bias_7, dropout_p=0.0
        )
        q_3 = k_3 = v_3 = attn_bias_7 = None
        view_20 = x_66.view(1, 12, 196, -1)
        x_66 = None
        transpose_3 = view_20.transpose(1, 2)
        view_20 = None
        x_67 = transpose_3.reshape(1, 196, -1)
        transpose_3 = None
        x_68 = torch._C._nn.linear(
            x_67,
            l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_,
        )
        x_67 = (
            l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_69 = torch.nn.functional.dropout(x_68, 0.0, False, False)
        x_68 = None
        x_70 = x_69.view(1, 14, 14, -1)
        x_69 = None
        x_71 = x_70.view(1, 1, 1, 14, 14, -1)
        x_70 = None
        permute_16 = x_71.permute(0, 1, 3, 2, 4, 5)
        x_71 = None
        contiguous_7 = permute_16.contiguous()
        permute_16 = None
        x_72 = contiguous_7.view(1, 14, 14, -1)
        contiguous_7 = None
        getitem_50 = x_72[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_72 = None
        x_73 = getitem_50.contiguous()
        getitem_50 = None
        x_74 = x_61 + x_73
        x_61 = x_73 = None
        x_75 = x_74.reshape(1, 196, -1)
        x_74 = None
        layer_norm_7 = torch.nn.functional.layer_norm(
            x_75,
            (768,),
            l_self_modules_blocks_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_3_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_3_modules_norm2_parameters_bias_
        ) = None
        x_76 = torch._C._nn.linear(
            layer_norm_7,
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_7 = (
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_77 = torch._C._nn.gelu(x_76, approximate="none")
        x_76 = None
        x_78 = torch.nn.functional.dropout(x_77, 0.0, False, False)
        x_77 = None
        x_79 = torch._C._nn.linear(
            x_78,
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_78 = (
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_80 = torch.nn.functional.dropout(x_79, 0.0, False, False)
        x_79 = None
        x_81 = x_75 + x_80
        x_75 = x_80 = None
        x_82 = x_81.reshape(1, 14, 14, -1)
        x_81 = None
        x_83 = torch.nn.functional.layer_norm(
            x_82,
            (768,),
            l_self_modules_blocks_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_4_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_4_modules_norm1_parameters_bias_
        ) = None
        x_84 = torch._C._nn.pad(x_83, (0, 0, 0, 0, 0, 0), "constant", None)
        x_83 = None
        x_85 = x_84.view(1, 1, 14, 1, 14, 768)
        x_84 = None
        permute_17 = x_85.permute(0, 1, 3, 2, 4, 5)
        x_85 = None
        contiguous_9 = permute_17.contiguous()
        permute_17 = None
        windows_3 = contiguous_9.view(-1, 14, 14, 768)
        contiguous_9 = None
        x_86 = windows_3.reshape(1, 196, -1)
        windows_3 = None
        linear_16 = torch._C._nn.linear(
            x_86,
            l_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_bias_,
        )
        x_86 = (
            l_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_26 = linear_16.view(1, 196, 3, 12, -1)
        linear_16 = None
        qkv_4 = view_26.permute(2, 0, 3, 1, 4)
        view_26 = None
        reshape_34 = qkv_4.reshape(3, 12, 196, -1)
        qkv_4 = None
        unbind_4 = reshape_34.unbind(0)
        reshape_34 = None
        q_4 = unbind_4[0]
        k_4 = unbind_4[1]
        v_4 = unbind_4[2]
        unbind_4 = None
        arange_16 = torch.arange(14)
        getitem_54 = arange_16[(slice(None, None, None), None)]
        arange_16 = None
        q_coords_8 = getitem_54 * 1.0
        getitem_54 = None
        arange_17 = torch.arange(14)
        getitem_55 = arange_17[(None, slice(None, None, None))]
        arange_17 = None
        k_coords_8 = getitem_55 * 1.0
        getitem_55 = None
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
        getitem_57 = arange_18[(slice(None, None, None), None)]
        arange_18 = None
        q_coords_9 = getitem_57 * 1.0
        getitem_57 = None
        arange_19 = torch.arange(14)
        getitem_58 = arange_19[(None, slice(None, None, None))]
        arange_19 = None
        k_coords_9 = getitem_58 * 1.0
        getitem_58 = None
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
        r_q_4 = q_4.reshape(12, 14, 14, 64)
        rel_h_4 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_4, Rh_4)
        Rh_4 = None
        rel_w_4 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_4, Rw_4)
        r_q_4 = Rw_4 = None
        getitem_60 = rel_h_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_4 = None
        getitem_61 = rel_w_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_4 = None
        attn_bias_8 = getitem_60 + getitem_61
        getitem_60 = getitem_61 = None
        attn_bias_9 = attn_bias_8.reshape(-1, 196, 196)
        attn_bias_8 = None
        x_87 = torch._C._nn.scaled_dot_product_attention(
            q_4, k_4, v_4, attn_mask=attn_bias_9, dropout_p=0.0
        )
        q_4 = k_4 = v_4 = attn_bias_9 = None
        view_27 = x_87.view(1, 12, 196, -1)
        x_87 = None
        transpose_4 = view_27.transpose(1, 2)
        view_27 = None
        x_88 = transpose_4.reshape(1, 196, -1)
        transpose_4 = None
        x_89 = torch._C._nn.linear(
            x_88,
            l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_,
        )
        x_88 = (
            l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_90 = torch.nn.functional.dropout(x_89, 0.0, False, False)
        x_89 = None
        x_91 = x_90.view(1, 14, 14, -1)
        x_90 = None
        x_92 = x_91.view(1, 1, 1, 14, 14, -1)
        x_91 = None
        permute_19 = x_92.permute(0, 1, 3, 2, 4, 5)
        x_92 = None
        contiguous_10 = permute_19.contiguous()
        permute_19 = None
        x_93 = contiguous_10.view(1, 14, 14, -1)
        contiguous_10 = None
        getitem_62 = x_93[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_93 = None
        x_94 = getitem_62.contiguous()
        getitem_62 = None
        x_95 = x_82 + x_94
        x_82 = x_94 = None
        x_96 = x_95.reshape(1, 196, -1)
        x_95 = None
        layer_norm_9 = torch.nn.functional.layer_norm(
            x_96,
            (768,),
            l_self_modules_blocks_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_4_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_4_modules_norm2_parameters_bias_
        ) = None
        x_97 = torch._C._nn.linear(
            layer_norm_9,
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_9 = (
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_98 = torch._C._nn.gelu(x_97, approximate="none")
        x_97 = None
        x_99 = torch.nn.functional.dropout(x_98, 0.0, False, False)
        x_98 = None
        x_100 = torch._C._nn.linear(
            x_99,
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_99 = (
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_101 = torch.nn.functional.dropout(x_100, 0.0, False, False)
        x_100 = None
        x_102 = x_96 + x_101
        x_96 = x_101 = None
        x_103 = x_102.reshape(1, 14, 14, -1)
        x_102 = None
        x_104 = torch.nn.functional.layer_norm(
            x_103,
            (768,),
            l_self_modules_blocks_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_5_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_5_modules_norm1_parameters_bias_
        ) = None
        x_105 = x_104.reshape(1, 196, -1)
        x_104 = None
        linear_20 = torch._C._nn.linear(
            x_105,
            l_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_bias_,
        )
        x_105 = (
            l_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_31 = linear_20.view(1, 196, 3, 12, -1)
        linear_20 = None
        qkv_5 = view_31.permute(2, 0, 3, 1, 4)
        view_31 = None
        reshape_41 = qkv_5.reshape(3, 12, 196, -1)
        qkv_5 = None
        unbind_5 = reshape_41.unbind(0)
        reshape_41 = None
        q_5 = unbind_5[0]
        k_5 = unbind_5[1]
        v_5 = unbind_5[2]
        unbind_5 = None
        reshape_42 = (
            l_self_modules_blocks_modules_5_modules_attn_parameters_rel_pos_h_.reshape(
                1, 127, -1
            )
        )
        l_self_modules_blocks_modules_5_modules_attn_parameters_rel_pos_h_ = None
        permute_21 = reshape_42.permute(0, 2, 1)
        reshape_42 = None
        rel_pos_resized_4 = torch.nn.functional.interpolate(
            permute_21, size=27, mode="linear"
        )
        permute_21 = None
        reshape_43 = rel_pos_resized_4.reshape(-1, 27)
        rel_pos_resized_4 = None
        rel_pos_resized_5 = reshape_43.permute(1, 0)
        reshape_43 = None
        arange_20 = torch.arange(14)
        getitem_66 = arange_20[(slice(None, None, None), None)]
        arange_20 = None
        q_coords_10 = getitem_66 * 1.0
        getitem_66 = None
        arange_21 = torch.arange(14)
        getitem_67 = arange_21[(None, slice(None, None, None))]
        arange_21 = None
        k_coords_10 = getitem_67 * 1.0
        getitem_67 = None
        sub_10 = q_coords_10 - k_coords_10
        q_coords_10 = k_coords_10 = None
        relative_coords_10 = sub_10 + 13.0
        sub_10 = None
        long_10 = relative_coords_10.long()
        relative_coords_10 = None
        Rh_5 = rel_pos_resized_5[long_10]
        rel_pos_resized_5 = long_10 = None
        reshape_44 = (
            l_self_modules_blocks_modules_5_modules_attn_parameters_rel_pos_w_.reshape(
                1, 127, -1
            )
        )
        l_self_modules_blocks_modules_5_modules_attn_parameters_rel_pos_w_ = None
        permute_23 = reshape_44.permute(0, 2, 1)
        reshape_44 = None
        rel_pos_resized_6 = torch.nn.functional.interpolate(
            permute_23, size=27, mode="linear"
        )
        permute_23 = None
        reshape_45 = rel_pos_resized_6.reshape(-1, 27)
        rel_pos_resized_6 = None
        rel_pos_resized_7 = reshape_45.permute(1, 0)
        reshape_45 = None
        arange_22 = torch.arange(14)
        getitem_69 = arange_22[(slice(None, None, None), None)]
        arange_22 = None
        q_coords_11 = getitem_69 * 1.0
        getitem_69 = None
        arange_23 = torch.arange(14)
        getitem_70 = arange_23[(None, slice(None, None, None))]
        arange_23 = None
        k_coords_11 = getitem_70 * 1.0
        getitem_70 = None
        sub_11 = q_coords_11 - k_coords_11
        q_coords_11 = k_coords_11 = None
        relative_coords_11 = sub_11 + 13.0
        sub_11 = None
        long_11 = relative_coords_11.long()
        relative_coords_11 = None
        Rw_5 = rel_pos_resized_7[long_11]
        rel_pos_resized_7 = long_11 = None
        r_q_5 = q_5.reshape(12, 14, 14, 64)
        rel_h_5 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_5, Rh_5)
        Rh_5 = None
        rel_w_5 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_5, Rw_5)
        r_q_5 = Rw_5 = None
        getitem_72 = rel_h_5[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_5 = None
        getitem_73 = rel_w_5[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_5 = None
        attn_bias_10 = getitem_72 + getitem_73
        getitem_72 = getitem_73 = None
        attn_bias_11 = attn_bias_10.reshape(-1, 196, 196)
        attn_bias_10 = None
        x_106 = torch._C._nn.scaled_dot_product_attention(
            q_5, k_5, v_5, attn_mask=attn_bias_11, dropout_p=0.0
        )
        q_5 = k_5 = v_5 = attn_bias_11 = None
        view_32 = x_106.view(1, 12, 196, -1)
        x_106 = None
        transpose_5 = view_32.transpose(1, 2)
        view_32 = None
        x_107 = transpose_5.reshape(1, 196, -1)
        transpose_5 = None
        x_108 = torch._C._nn.linear(
            x_107,
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_,
        )
        x_107 = (
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_109 = torch.nn.functional.dropout(x_108, 0.0, False, False)
        x_108 = None
        x_110 = x_109.view(1, 14, 14, -1)
        x_109 = None
        x_111 = x_103 + x_110
        x_103 = x_110 = None
        x_112 = x_111.reshape(1, 196, -1)
        x_111 = None
        layer_norm_11 = torch.nn.functional.layer_norm(
            x_112,
            (768,),
            l_self_modules_blocks_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_5_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_5_modules_norm2_parameters_bias_
        ) = None
        x_113 = torch._C._nn.linear(
            layer_norm_11,
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_11 = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_114 = torch._C._nn.gelu(x_113, approximate="none")
        x_113 = None
        x_115 = torch.nn.functional.dropout(x_114, 0.0, False, False)
        x_114 = None
        x_116 = torch._C._nn.linear(
            x_115,
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_115 = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_117 = torch.nn.functional.dropout(x_116, 0.0, False, False)
        x_116 = None
        x_118 = x_112 + x_117
        x_112 = x_117 = None
        x_119 = x_118.reshape(1, 14, 14, -1)
        x_118 = None
        x_120 = torch.nn.functional.layer_norm(
            x_119,
            (768,),
            l_self_modules_blocks_modules_6_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_6_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_6_modules_norm1_parameters_bias_
        ) = None
        x_121 = torch._C._nn.pad(x_120, (0, 0, 0, 0, 0, 0), "constant", None)
        x_120 = None
        x_122 = x_121.view(1, 1, 14, 1, 14, 768)
        x_121 = None
        permute_25 = x_122.permute(0, 1, 3, 2, 4, 5)
        x_122 = None
        contiguous_12 = permute_25.contiguous()
        permute_25 = None
        windows_4 = contiguous_12.view(-1, 14, 14, 768)
        contiguous_12 = None
        x_123 = windows_4.reshape(1, 196, -1)
        windows_4 = None
        linear_24 = torch._C._nn.linear(
            x_123,
            l_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_bias_,
        )
        x_123 = (
            l_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_36 = linear_24.view(1, 196, 3, 12, -1)
        linear_24 = None
        qkv_6 = view_36.permute(2, 0, 3, 1, 4)
        view_36 = None
        reshape_52 = qkv_6.reshape(3, 12, 196, -1)
        qkv_6 = None
        unbind_6 = reshape_52.unbind(0)
        reshape_52 = None
        q_6 = unbind_6[0]
        k_6 = unbind_6[1]
        v_6 = unbind_6[2]
        unbind_6 = None
        arange_24 = torch.arange(14)
        getitem_77 = arange_24[(slice(None, None, None), None)]
        arange_24 = None
        q_coords_12 = getitem_77 * 1.0
        getitem_77 = None
        arange_25 = torch.arange(14)
        getitem_78 = arange_25[(None, slice(None, None, None))]
        arange_25 = None
        k_coords_12 = getitem_78 * 1.0
        getitem_78 = None
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
        getitem_80 = arange_26[(slice(None, None, None), None)]
        arange_26 = None
        q_coords_13 = getitem_80 * 1.0
        getitem_80 = None
        arange_27 = torch.arange(14)
        getitem_81 = arange_27[(None, slice(None, None, None))]
        arange_27 = None
        k_coords_13 = getitem_81 * 1.0
        getitem_81 = None
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
        r_q_6 = q_6.reshape(12, 14, 14, 64)
        rel_h_6 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_6, Rh_6)
        Rh_6 = None
        rel_w_6 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_6, Rw_6)
        r_q_6 = Rw_6 = None
        getitem_83 = rel_h_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_6 = None
        getitem_84 = rel_w_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_6 = None
        attn_bias_12 = getitem_83 + getitem_84
        getitem_83 = getitem_84 = None
        attn_bias_13 = attn_bias_12.reshape(-1, 196, 196)
        attn_bias_12 = None
        x_124 = torch._C._nn.scaled_dot_product_attention(
            q_6, k_6, v_6, attn_mask=attn_bias_13, dropout_p=0.0
        )
        q_6 = k_6 = v_6 = attn_bias_13 = None
        view_37 = x_124.view(1, 12, 196, -1)
        x_124 = None
        transpose_6 = view_37.transpose(1, 2)
        view_37 = None
        x_125 = transpose_6.reshape(1, 196, -1)
        transpose_6 = None
        x_126 = torch._C._nn.linear(
            x_125,
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_,
        )
        x_125 = (
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_127 = torch.nn.functional.dropout(x_126, 0.0, False, False)
        x_126 = None
        x_128 = x_127.view(1, 14, 14, -1)
        x_127 = None
        x_129 = x_128.view(1, 1, 1, 14, 14, -1)
        x_128 = None
        permute_27 = x_129.permute(0, 1, 3, 2, 4, 5)
        x_129 = None
        contiguous_13 = permute_27.contiguous()
        permute_27 = None
        x_130 = contiguous_13.view(1, 14, 14, -1)
        contiguous_13 = None
        getitem_85 = x_130[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_130 = None
        x_131 = getitem_85.contiguous()
        getitem_85 = None
        x_132 = x_119 + x_131
        x_119 = x_131 = None
        x_133 = x_132.reshape(1, 196, -1)
        x_132 = None
        layer_norm_13 = torch.nn.functional.layer_norm(
            x_133,
            (768,),
            l_self_modules_blocks_modules_6_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_6_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_6_modules_norm2_parameters_bias_
        ) = None
        x_134 = torch._C._nn.linear(
            layer_norm_13,
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_13 = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_135 = torch._C._nn.gelu(x_134, approximate="none")
        x_134 = None
        x_136 = torch.nn.functional.dropout(x_135, 0.0, False, False)
        x_135 = None
        x_137 = torch._C._nn.linear(
            x_136,
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_136 = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_138 = torch.nn.functional.dropout(x_137, 0.0, False, False)
        x_137 = None
        x_139 = x_133 + x_138
        x_133 = x_138 = None
        x_140 = x_139.reshape(1, 14, 14, -1)
        x_139 = None
        x_141 = torch.nn.functional.layer_norm(
            x_140,
            (768,),
            l_self_modules_blocks_modules_7_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_7_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_7_modules_norm1_parameters_bias_
        ) = None
        x_142 = torch._C._nn.pad(x_141, (0, 0, 0, 0, 0, 0), "constant", None)
        x_141 = None
        x_143 = x_142.view(1, 1, 14, 1, 14, 768)
        x_142 = None
        permute_28 = x_143.permute(0, 1, 3, 2, 4, 5)
        x_143 = None
        contiguous_15 = permute_28.contiguous()
        permute_28 = None
        windows_5 = contiguous_15.view(-1, 14, 14, 768)
        contiguous_15 = None
        x_144 = windows_5.reshape(1, 196, -1)
        windows_5 = None
        linear_28 = torch._C._nn.linear(
            x_144,
            l_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_bias_,
        )
        x_144 = (
            l_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_43 = linear_28.view(1, 196, 3, 12, -1)
        linear_28 = None
        qkv_7 = view_43.permute(2, 0, 3, 1, 4)
        view_43 = None
        reshape_59 = qkv_7.reshape(3, 12, 196, -1)
        qkv_7 = None
        unbind_7 = reshape_59.unbind(0)
        reshape_59 = None
        q_7 = unbind_7[0]
        k_7 = unbind_7[1]
        v_7 = unbind_7[2]
        unbind_7 = None
        arange_28 = torch.arange(14)
        getitem_89 = arange_28[(slice(None, None, None), None)]
        arange_28 = None
        q_coords_14 = getitem_89 * 1.0
        getitem_89 = None
        arange_29 = torch.arange(14)
        getitem_90 = arange_29[(None, slice(None, None, None))]
        arange_29 = None
        k_coords_14 = getitem_90 * 1.0
        getitem_90 = None
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
        getitem_92 = arange_30[(slice(None, None, None), None)]
        arange_30 = None
        q_coords_15 = getitem_92 * 1.0
        getitem_92 = None
        arange_31 = torch.arange(14)
        getitem_93 = arange_31[(None, slice(None, None, None))]
        arange_31 = None
        k_coords_15 = getitem_93 * 1.0
        getitem_93 = None
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
        r_q_7 = q_7.reshape(12, 14, 14, 64)
        rel_h_7 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_7, Rh_7)
        Rh_7 = None
        rel_w_7 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_7, Rw_7)
        r_q_7 = Rw_7 = None
        getitem_95 = rel_h_7[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_7 = None
        getitem_96 = rel_w_7[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_7 = None
        attn_bias_14 = getitem_95 + getitem_96
        getitem_95 = getitem_96 = None
        attn_bias_15 = attn_bias_14.reshape(-1, 196, 196)
        attn_bias_14 = None
        x_145 = torch._C._nn.scaled_dot_product_attention(
            q_7, k_7, v_7, attn_mask=attn_bias_15, dropout_p=0.0
        )
        q_7 = k_7 = v_7 = attn_bias_15 = None
        view_44 = x_145.view(1, 12, 196, -1)
        x_145 = None
        transpose_7 = view_44.transpose(1, 2)
        view_44 = None
        x_146 = transpose_7.reshape(1, 196, -1)
        transpose_7 = None
        x_147 = torch._C._nn.linear(
            x_146,
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_,
        )
        x_146 = (
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_148 = torch.nn.functional.dropout(x_147, 0.0, False, False)
        x_147 = None
        x_149 = x_148.view(1, 14, 14, -1)
        x_148 = None
        x_150 = x_149.view(1, 1, 1, 14, 14, -1)
        x_149 = None
        permute_30 = x_150.permute(0, 1, 3, 2, 4, 5)
        x_150 = None
        contiguous_16 = permute_30.contiguous()
        permute_30 = None
        x_151 = contiguous_16.view(1, 14, 14, -1)
        contiguous_16 = None
        getitem_97 = x_151[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_151 = None
        x_152 = getitem_97.contiguous()
        getitem_97 = None
        x_153 = x_140 + x_152
        x_140 = x_152 = None
        x_154 = x_153.reshape(1, 196, -1)
        x_153 = None
        layer_norm_15 = torch.nn.functional.layer_norm(
            x_154,
            (768,),
            l_self_modules_blocks_modules_7_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_7_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_7_modules_norm2_parameters_bias_
        ) = None
        x_155 = torch._C._nn.linear(
            layer_norm_15,
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_15 = (
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_156 = torch._C._nn.gelu(x_155, approximate="none")
        x_155 = None
        x_157 = torch.nn.functional.dropout(x_156, 0.0, False, False)
        x_156 = None
        x_158 = torch._C._nn.linear(
            x_157,
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_157 = (
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_159 = torch.nn.functional.dropout(x_158, 0.0, False, False)
        x_158 = None
        x_160 = x_154 + x_159
        x_154 = x_159 = None
        x_161 = x_160.reshape(1, 14, 14, -1)
        x_160 = None
        x_162 = torch.nn.functional.layer_norm(
            x_161,
            (768,),
            l_self_modules_blocks_modules_8_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_8_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_8_modules_norm1_parameters_bias_
        ) = None
        x_163 = x_162.reshape(1, 196, -1)
        x_162 = None
        linear_32 = torch._C._nn.linear(
            x_163,
            l_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_bias_,
        )
        x_163 = (
            l_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_48 = linear_32.view(1, 196, 3, 12, -1)
        linear_32 = None
        qkv_8 = view_48.permute(2, 0, 3, 1, 4)
        view_48 = None
        reshape_66 = qkv_8.reshape(3, 12, 196, -1)
        qkv_8 = None
        unbind_8 = reshape_66.unbind(0)
        reshape_66 = None
        q_8 = unbind_8[0]
        k_8 = unbind_8[1]
        v_8 = unbind_8[2]
        unbind_8 = None
        reshape_67 = (
            l_self_modules_blocks_modules_8_modules_attn_parameters_rel_pos_h_.reshape(
                1, 127, -1
            )
        )
        l_self_modules_blocks_modules_8_modules_attn_parameters_rel_pos_h_ = None
        permute_32 = reshape_67.permute(0, 2, 1)
        reshape_67 = None
        rel_pos_resized_8 = torch.nn.functional.interpolate(
            permute_32, size=27, mode="linear"
        )
        permute_32 = None
        reshape_68 = rel_pos_resized_8.reshape(-1, 27)
        rel_pos_resized_8 = None
        rel_pos_resized_9 = reshape_68.permute(1, 0)
        reshape_68 = None
        arange_32 = torch.arange(14)
        getitem_101 = arange_32[(slice(None, None, None), None)]
        arange_32 = None
        q_coords_16 = getitem_101 * 1.0
        getitem_101 = None
        arange_33 = torch.arange(14)
        getitem_102 = arange_33[(None, slice(None, None, None))]
        arange_33 = None
        k_coords_16 = getitem_102 * 1.0
        getitem_102 = None
        sub_16 = q_coords_16 - k_coords_16
        q_coords_16 = k_coords_16 = None
        relative_coords_16 = sub_16 + 13.0
        sub_16 = None
        long_16 = relative_coords_16.long()
        relative_coords_16 = None
        Rh_8 = rel_pos_resized_9[long_16]
        rel_pos_resized_9 = long_16 = None
        reshape_69 = (
            l_self_modules_blocks_modules_8_modules_attn_parameters_rel_pos_w_.reshape(
                1, 127, -1
            )
        )
        l_self_modules_blocks_modules_8_modules_attn_parameters_rel_pos_w_ = None
        permute_34 = reshape_69.permute(0, 2, 1)
        reshape_69 = None
        rel_pos_resized_10 = torch.nn.functional.interpolate(
            permute_34, size=27, mode="linear"
        )
        permute_34 = None
        reshape_70 = rel_pos_resized_10.reshape(-1, 27)
        rel_pos_resized_10 = None
        rel_pos_resized_11 = reshape_70.permute(1, 0)
        reshape_70 = None
        arange_34 = torch.arange(14)
        getitem_104 = arange_34[(slice(None, None, None), None)]
        arange_34 = None
        q_coords_17 = getitem_104 * 1.0
        getitem_104 = None
        arange_35 = torch.arange(14)
        getitem_105 = arange_35[(None, slice(None, None, None))]
        arange_35 = None
        k_coords_17 = getitem_105 * 1.0
        getitem_105 = None
        sub_17 = q_coords_17 - k_coords_17
        q_coords_17 = k_coords_17 = None
        relative_coords_17 = sub_17 + 13.0
        sub_17 = None
        long_17 = relative_coords_17.long()
        relative_coords_17 = None
        Rw_8 = rel_pos_resized_11[long_17]
        rel_pos_resized_11 = long_17 = None
        r_q_8 = q_8.reshape(12, 14, 14, 64)
        rel_h_8 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_8, Rh_8)
        Rh_8 = None
        rel_w_8 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_8, Rw_8)
        r_q_8 = Rw_8 = None
        getitem_107 = rel_h_8[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_8 = None
        getitem_108 = rel_w_8[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_8 = None
        attn_bias_16 = getitem_107 + getitem_108
        getitem_107 = getitem_108 = None
        attn_bias_17 = attn_bias_16.reshape(-1, 196, 196)
        attn_bias_16 = None
        x_164 = torch._C._nn.scaled_dot_product_attention(
            q_8, k_8, v_8, attn_mask=attn_bias_17, dropout_p=0.0
        )
        q_8 = k_8 = v_8 = attn_bias_17 = None
        view_49 = x_164.view(1, 12, 196, -1)
        x_164 = None
        transpose_8 = view_49.transpose(1, 2)
        view_49 = None
        x_165 = transpose_8.reshape(1, 196, -1)
        transpose_8 = None
        x_166 = torch._C._nn.linear(
            x_165,
            l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_,
        )
        x_165 = (
            l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_167 = torch.nn.functional.dropout(x_166, 0.0, False, False)
        x_166 = None
        x_168 = x_167.view(1, 14, 14, -1)
        x_167 = None
        x_169 = x_161 + x_168
        x_161 = x_168 = None
        x_170 = x_169.reshape(1, 196, -1)
        x_169 = None
        layer_norm_17 = torch.nn.functional.layer_norm(
            x_170,
            (768,),
            l_self_modules_blocks_modules_8_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_8_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_8_modules_norm2_parameters_bias_
        ) = None
        x_171 = torch._C._nn.linear(
            layer_norm_17,
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_17 = (
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_172 = torch._C._nn.gelu(x_171, approximate="none")
        x_171 = None
        x_173 = torch.nn.functional.dropout(x_172, 0.0, False, False)
        x_172 = None
        x_174 = torch._C._nn.linear(
            x_173,
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_173 = (
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_175 = torch.nn.functional.dropout(x_174, 0.0, False, False)
        x_174 = None
        x_176 = x_170 + x_175
        x_170 = x_175 = None
        x_177 = x_176.reshape(1, 14, 14, -1)
        x_176 = None
        x_178 = torch.nn.functional.layer_norm(
            x_177,
            (768,),
            l_self_modules_blocks_modules_9_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_9_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_9_modules_norm1_parameters_bias_
        ) = None
        x_179 = torch._C._nn.pad(x_178, (0, 0, 0, 0, 0, 0), "constant", None)
        x_178 = None
        x_180 = x_179.view(1, 1, 14, 1, 14, 768)
        x_179 = None
        permute_36 = x_180.permute(0, 1, 3, 2, 4, 5)
        x_180 = None
        contiguous_18 = permute_36.contiguous()
        permute_36 = None
        windows_6 = contiguous_18.view(-1, 14, 14, 768)
        contiguous_18 = None
        x_181 = windows_6.reshape(1, 196, -1)
        windows_6 = None
        linear_36 = torch._C._nn.linear(
            x_181,
            l_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_bias_,
        )
        x_181 = (
            l_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_53 = linear_36.view(1, 196, 3, 12, -1)
        linear_36 = None
        qkv_9 = view_53.permute(2, 0, 3, 1, 4)
        view_53 = None
        reshape_77 = qkv_9.reshape(3, 12, 196, -1)
        qkv_9 = None
        unbind_9 = reshape_77.unbind(0)
        reshape_77 = None
        q_9 = unbind_9[0]
        k_9 = unbind_9[1]
        v_9 = unbind_9[2]
        unbind_9 = None
        arange_36 = torch.arange(14)
        getitem_112 = arange_36[(slice(None, None, None), None)]
        arange_36 = None
        q_coords_18 = getitem_112 * 1.0
        getitem_112 = None
        arange_37 = torch.arange(14)
        getitem_113 = arange_37[(None, slice(None, None, None))]
        arange_37 = None
        k_coords_18 = getitem_113 * 1.0
        getitem_113 = None
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
        getitem_115 = arange_38[(slice(None, None, None), None)]
        arange_38 = None
        q_coords_19 = getitem_115 * 1.0
        getitem_115 = None
        arange_39 = torch.arange(14)
        getitem_116 = arange_39[(None, slice(None, None, None))]
        arange_39 = None
        k_coords_19 = getitem_116 * 1.0
        getitem_116 = None
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
        r_q_9 = q_9.reshape(12, 14, 14, 64)
        rel_h_9 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_9, Rh_9)
        Rh_9 = None
        rel_w_9 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_9, Rw_9)
        r_q_9 = Rw_9 = None
        getitem_118 = rel_h_9[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_9 = None
        getitem_119 = rel_w_9[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_9 = None
        attn_bias_18 = getitem_118 + getitem_119
        getitem_118 = getitem_119 = None
        attn_bias_19 = attn_bias_18.reshape(-1, 196, 196)
        attn_bias_18 = None
        x_182 = torch._C._nn.scaled_dot_product_attention(
            q_9, k_9, v_9, attn_mask=attn_bias_19, dropout_p=0.0
        )
        q_9 = k_9 = v_9 = attn_bias_19 = None
        view_54 = x_182.view(1, 12, 196, -1)
        x_182 = None
        transpose_9 = view_54.transpose(1, 2)
        view_54 = None
        x_183 = transpose_9.reshape(1, 196, -1)
        transpose_9 = None
        x_184 = torch._C._nn.linear(
            x_183,
            l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_,
        )
        x_183 = (
            l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_185 = torch.nn.functional.dropout(x_184, 0.0, False, False)
        x_184 = None
        x_186 = x_185.view(1, 14, 14, -1)
        x_185 = None
        x_187 = x_186.view(1, 1, 1, 14, 14, -1)
        x_186 = None
        permute_38 = x_187.permute(0, 1, 3, 2, 4, 5)
        x_187 = None
        contiguous_19 = permute_38.contiguous()
        permute_38 = None
        x_188 = contiguous_19.view(1, 14, 14, -1)
        contiguous_19 = None
        getitem_120 = x_188[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_188 = None
        x_189 = getitem_120.contiguous()
        getitem_120 = None
        x_190 = x_177 + x_189
        x_177 = x_189 = None
        x_191 = x_190.reshape(1, 196, -1)
        x_190 = None
        layer_norm_19 = torch.nn.functional.layer_norm(
            x_191,
            (768,),
            l_self_modules_blocks_modules_9_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_9_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_9_modules_norm2_parameters_bias_
        ) = None
        x_192 = torch._C._nn.linear(
            layer_norm_19,
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_19 = (
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_193 = torch._C._nn.gelu(x_192, approximate="none")
        x_192 = None
        x_194 = torch.nn.functional.dropout(x_193, 0.0, False, False)
        x_193 = None
        x_195 = torch._C._nn.linear(
            x_194,
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_194 = (
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_196 = torch.nn.functional.dropout(x_195, 0.0, False, False)
        x_195 = None
        x_197 = x_191 + x_196
        x_191 = x_196 = None
        x_198 = x_197.reshape(1, 14, 14, -1)
        x_197 = None
        x_199 = torch.nn.functional.layer_norm(
            x_198,
            (768,),
            l_self_modules_blocks_modules_10_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_10_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_10_modules_norm1_parameters_bias_
        ) = None
        x_200 = torch._C._nn.pad(x_199, (0, 0, 0, 0, 0, 0), "constant", None)
        x_199 = None
        x_201 = x_200.view(1, 1, 14, 1, 14, 768)
        x_200 = None
        permute_39 = x_201.permute(0, 1, 3, 2, 4, 5)
        x_201 = None
        contiguous_21 = permute_39.contiguous()
        permute_39 = None
        windows_7 = contiguous_21.view(-1, 14, 14, 768)
        contiguous_21 = None
        x_202 = windows_7.reshape(1, 196, -1)
        windows_7 = None
        linear_40 = torch._C._nn.linear(
            x_202,
            l_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_bias_,
        )
        x_202 = (
            l_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_60 = linear_40.view(1, 196, 3, 12, -1)
        linear_40 = None
        qkv_10 = view_60.permute(2, 0, 3, 1, 4)
        view_60 = None
        reshape_84 = qkv_10.reshape(3, 12, 196, -1)
        qkv_10 = None
        unbind_10 = reshape_84.unbind(0)
        reshape_84 = None
        q_10 = unbind_10[0]
        k_10 = unbind_10[1]
        v_10 = unbind_10[2]
        unbind_10 = None
        arange_40 = torch.arange(14)
        getitem_124 = arange_40[(slice(None, None, None), None)]
        arange_40 = None
        q_coords_20 = getitem_124 * 1.0
        getitem_124 = None
        arange_41 = torch.arange(14)
        getitem_125 = arange_41[(None, slice(None, None, None))]
        arange_41 = None
        k_coords_20 = getitem_125 * 1.0
        getitem_125 = None
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
        getitem_127 = arange_42[(slice(None, None, None), None)]
        arange_42 = None
        q_coords_21 = getitem_127 * 1.0
        getitem_127 = None
        arange_43 = torch.arange(14)
        getitem_128 = arange_43[(None, slice(None, None, None))]
        arange_43 = None
        k_coords_21 = getitem_128 * 1.0
        getitem_128 = None
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
        r_q_10 = q_10.reshape(12, 14, 14, 64)
        rel_h_10 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_10, Rh_10)
        Rh_10 = None
        rel_w_10 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_10, Rw_10)
        r_q_10 = Rw_10 = None
        getitem_130 = rel_h_10[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_10 = None
        getitem_131 = rel_w_10[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_10 = None
        attn_bias_20 = getitem_130 + getitem_131
        getitem_130 = getitem_131 = None
        attn_bias_21 = attn_bias_20.reshape(-1, 196, 196)
        attn_bias_20 = None
        x_203 = torch._C._nn.scaled_dot_product_attention(
            q_10, k_10, v_10, attn_mask=attn_bias_21, dropout_p=0.0
        )
        q_10 = k_10 = v_10 = attn_bias_21 = None
        view_61 = x_203.view(1, 12, 196, -1)
        x_203 = None
        transpose_10 = view_61.transpose(1, 2)
        view_61 = None
        x_204 = transpose_10.reshape(1, 196, -1)
        transpose_10 = None
        x_205 = torch._C._nn.linear(
            x_204,
            l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_,
        )
        x_204 = l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_206 = torch.nn.functional.dropout(x_205, 0.0, False, False)
        x_205 = None
        x_207 = x_206.view(1, 14, 14, -1)
        x_206 = None
        x_208 = x_207.view(1, 1, 1, 14, 14, -1)
        x_207 = None
        permute_41 = x_208.permute(0, 1, 3, 2, 4, 5)
        x_208 = None
        contiguous_22 = permute_41.contiguous()
        permute_41 = None
        x_209 = contiguous_22.view(1, 14, 14, -1)
        contiguous_22 = None
        getitem_132 = x_209[
            (
                slice(None, None, None),
                slice(None, 14, None),
                slice(None, 14, None),
                slice(None, None, None),
            )
        ]
        x_209 = None
        x_210 = getitem_132.contiguous()
        getitem_132 = None
        x_211 = x_198 + x_210
        x_198 = x_210 = None
        x_212 = x_211.reshape(1, 196, -1)
        x_211 = None
        layer_norm_21 = torch.nn.functional.layer_norm(
            x_212,
            (768,),
            l_self_modules_blocks_modules_10_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_10_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_10_modules_norm2_parameters_bias_
        ) = None
        x_213 = torch._C._nn.linear(
            layer_norm_21,
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_21 = (
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_214 = torch._C._nn.gelu(x_213, approximate="none")
        x_213 = None
        x_215 = torch.nn.functional.dropout(x_214, 0.0, False, False)
        x_214 = None
        x_216 = torch._C._nn.linear(
            x_215,
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_215 = (
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_217 = torch.nn.functional.dropout(x_216, 0.0, False, False)
        x_216 = None
        x_218 = x_212 + x_217
        x_212 = x_217 = None
        x_219 = x_218.reshape(1, 14, 14, -1)
        x_218 = None
        x_220 = torch.nn.functional.layer_norm(
            x_219,
            (768,),
            l_self_modules_blocks_modules_11_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_norm1_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_11_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_11_modules_norm1_parameters_bias_
        ) = None
        x_221 = x_220.reshape(1, 196, -1)
        x_220 = None
        linear_44 = torch._C._nn.linear(
            x_221,
            l_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_bias_,
        )
        x_221 = (
            l_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_bias_
        ) = None
        view_65 = linear_44.view(1, 196, 3, 12, -1)
        linear_44 = None
        qkv_11 = view_65.permute(2, 0, 3, 1, 4)
        view_65 = None
        reshape_91 = qkv_11.reshape(3, 12, 196, -1)
        qkv_11 = None
        unbind_11 = reshape_91.unbind(0)
        reshape_91 = None
        q_11 = unbind_11[0]
        k_11 = unbind_11[1]
        v_11 = unbind_11[2]
        unbind_11 = None
        reshape_92 = (
            l_self_modules_blocks_modules_11_modules_attn_parameters_rel_pos_h_.reshape(
                1, 127, -1
            )
        )
        l_self_modules_blocks_modules_11_modules_attn_parameters_rel_pos_h_ = None
        permute_43 = reshape_92.permute(0, 2, 1)
        reshape_92 = None
        rel_pos_resized_12 = torch.nn.functional.interpolate(
            permute_43, size=27, mode="linear"
        )
        permute_43 = None
        reshape_93 = rel_pos_resized_12.reshape(-1, 27)
        rel_pos_resized_12 = None
        rel_pos_resized_13 = reshape_93.permute(1, 0)
        reshape_93 = None
        arange_44 = torch.arange(14)
        getitem_136 = arange_44[(slice(None, None, None), None)]
        arange_44 = None
        q_coords_22 = getitem_136 * 1.0
        getitem_136 = None
        arange_45 = torch.arange(14)
        getitem_137 = arange_45[(None, slice(None, None, None))]
        arange_45 = None
        k_coords_22 = getitem_137 * 1.0
        getitem_137 = None
        sub_22 = q_coords_22 - k_coords_22
        q_coords_22 = k_coords_22 = None
        relative_coords_22 = sub_22 + 13.0
        sub_22 = None
        long_22 = relative_coords_22.long()
        relative_coords_22 = None
        Rh_11 = rel_pos_resized_13[long_22]
        rel_pos_resized_13 = long_22 = None
        reshape_94 = (
            l_self_modules_blocks_modules_11_modules_attn_parameters_rel_pos_w_.reshape(
                1, 127, -1
            )
        )
        l_self_modules_blocks_modules_11_modules_attn_parameters_rel_pos_w_ = None
        permute_45 = reshape_94.permute(0, 2, 1)
        reshape_94 = None
        rel_pos_resized_14 = torch.nn.functional.interpolate(
            permute_45, size=27, mode="linear"
        )
        permute_45 = None
        reshape_95 = rel_pos_resized_14.reshape(-1, 27)
        rel_pos_resized_14 = None
        rel_pos_resized_15 = reshape_95.permute(1, 0)
        reshape_95 = None
        arange_46 = torch.arange(14)
        getitem_139 = arange_46[(slice(None, None, None), None)]
        arange_46 = None
        q_coords_23 = getitem_139 * 1.0
        getitem_139 = None
        arange_47 = torch.arange(14)
        getitem_140 = arange_47[(None, slice(None, None, None))]
        arange_47 = None
        k_coords_23 = getitem_140 * 1.0
        getitem_140 = None
        sub_23 = q_coords_23 - k_coords_23
        q_coords_23 = k_coords_23 = None
        relative_coords_23 = sub_23 + 13.0
        sub_23 = None
        long_23 = relative_coords_23.long()
        relative_coords_23 = None
        Rw_11 = rel_pos_resized_15[long_23]
        rel_pos_resized_15 = long_23 = None
        r_q_11 = q_11.reshape(12, 14, 14, 64)
        rel_h_11 = torch.functional.einsum("bhwc,hkc->bhwk", r_q_11, Rh_11)
        Rh_11 = None
        rel_w_11 = torch.functional.einsum("bhwc,wkc->bhwk", r_q_11, Rw_11)
        r_q_11 = Rw_11 = None
        getitem_142 = rel_h_11[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
            )
        ]
        rel_h_11 = None
        getitem_143 = rel_w_11[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
            )
        ]
        rel_w_11 = None
        attn_bias_22 = getitem_142 + getitem_143
        getitem_142 = getitem_143 = None
        attn_bias_23 = attn_bias_22.reshape(-1, 196, 196)
        attn_bias_22 = None
        x_222 = torch._C._nn.scaled_dot_product_attention(
            q_11, k_11, v_11, attn_mask=attn_bias_23, dropout_p=0.0
        )
        q_11 = k_11 = v_11 = attn_bias_23 = None
        view_66 = x_222.view(1, 12, 196, -1)
        x_222 = None
        transpose_11 = view_66.transpose(1, 2)
        view_66 = None
        x_223 = transpose_11.reshape(1, 196, -1)
        transpose_11 = None
        x_224 = torch._C._nn.linear(
            x_223,
            l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_,
        )
        x_223 = l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_225 = torch.nn.functional.dropout(x_224, 0.0, False, False)
        x_224 = None
        x_226 = x_225.view(1, 14, 14, -1)
        x_225 = None
        x_227 = x_219 + x_226
        x_219 = x_226 = None
        x_228 = x_227.reshape(1, 196, -1)
        x_227 = None
        layer_norm_23 = torch.nn.functional.layer_norm(
            x_228,
            (768,),
            l_self_modules_blocks_modules_11_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_norm2_parameters_bias_,
            1e-05,
        )
        l_self_modules_blocks_modules_11_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_11_modules_norm2_parameters_bias_
        ) = None
        x_229 = torch._C._nn.linear(
            layer_norm_23,
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_,
        )
        layer_norm_23 = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_230 = torch._C._nn.gelu(x_229, approximate="none")
        x_229 = None
        x_231 = torch.nn.functional.dropout(x_230, 0.0, False, False)
        x_230 = None
        x_232 = torch._C._nn.linear(
            x_231,
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_231 = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_233 = torch.nn.functional.dropout(x_232, 0.0, False, False)
        x_232 = None
        x_234 = x_228 + x_233
        x_228 = x_233 = None
        x_235 = x_234.reshape(1, 14, 14, -1)
        x_234 = None
        permute_47 = x_235.permute(0, 3, 1, 2)
        x_235 = None
        input_1 = torch.conv2d(
            permute_47,
            l_self_modules_neck_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        permute_47 = l_self_modules_neck_modules_0_parameters_weight_ = None
        x_236 = input_1.permute(0, 2, 3, 1)
        input_1 = None
        x_237 = torch.nn.functional.layer_norm(
            x_236,
            (256,),
            l_self_modules_neck_modules_1_parameters_weight_,
            l_self_modules_neck_modules_1_parameters_bias_,
            1e-06,
        )
        x_236 = (
            l_self_modules_neck_modules_1_parameters_weight_
        ) = l_self_modules_neck_modules_1_parameters_bias_ = None
        x_238 = x_237.permute(0, 3, 1, 2)
        x_237 = None
        input_2 = torch.conv2d(
            x_238,
            l_self_modules_neck_modules_2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_238 = l_self_modules_neck_modules_2_parameters_weight_ = None
        x_239 = input_2.permute(0, 2, 3, 1)
        input_2 = None
        x_240 = torch.nn.functional.layer_norm(
            x_239,
            (256,),
            l_self_modules_neck_modules_3_parameters_weight_,
            l_self_modules_neck_modules_3_parameters_bias_,
            1e-06,
        )
        x_239 = (
            l_self_modules_neck_modules_3_parameters_weight_
        ) = l_self_modules_neck_modules_3_parameters_bias_ = None
        x_241 = x_240.permute(0, 3, 1, 2)
        x_240 = None
        x_242 = torch.nn.functional.adaptive_avg_pool2d(x_241, 1)
        x_241 = None
        x_243 = x_242.flatten(1, -1)
        x_242 = None
        x_244 = torch.nn.functional.dropout(x_243, 0.0, False, False)
        x_243 = None
        return (x_244,)
