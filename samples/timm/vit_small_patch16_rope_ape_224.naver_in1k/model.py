import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_x_: torch.Tensor,
        L_self_modules_patch_embed_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_patch_embed_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_parameters_pos_embed_: torch.nn.parameter.Parameter,
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
        x_2 = x_1 + l_self_parameters_pos_embed_
        x_1 = l_self_parameters_pos_embed_ = None
        x_3 = torch.cat([expand, x_2], dim=1)
        expand = x_2 = None
        x_4 = torch.nn.functional.dropout(x_3, 0.0, False, False)
        x_3 = None
        x_5 = torch.nn.functional.layer_norm(
            x_4,
            (384,),
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
        reshape = qkv.reshape(1, 197, 3, 6, -1)
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
        reshape_1 = stack.reshape((1, 6, 196, 64))
        stack = None
        mul_1 = reshape_1 * sin_emb
        reshape_1 = sin_emb = None
        add_1 = mul + mul_1
        mul = mul_1 = None
        cat_1 = torch.cat([getitem_7, add_1], dim=2)
        getitem_7 = add_1 = None
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
        reshape_2 = stack_1.reshape((1, 6, 196, 64))
        stack_1 = None
        mul_3 = reshape_2 * sin_emb_1
        reshape_2 = sin_emb_1 = None
        add_2 = mul_2 + mul_3
        mul_2 = mul_3 = None
        cat_2 = torch.cat([getitem_13, add_2], dim=2)
        getitem_13 = add_2 = None
        k_1 = cat_2.type_as(v)
        cat_2 = None
        x_6 = torch._C._nn.scaled_dot_product_attention(
            q_1, k_1, v, attn_mask=None, dropout_p=0.0
        )
        q_1 = k_1 = v = None
        transpose_1 = x_6.transpose(1, 2)
        x_6 = None
        x_7 = transpose_1.reshape(1, 197, 384)
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
            (384,),
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
        x_18 = torch.nn.functional.layer_norm(
            x_17,
            (384,),
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
        reshape_4 = qkv_2.reshape(1, 197, 3, 6, -1)
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
        reshape_5 = stack_2.reshape((1, 6, 196, 64))
        stack_2 = None
        mul_7 = reshape_5 * sin_emb_2
        reshape_5 = sin_emb_2 = None
        add_5 = mul_6 + mul_7
        mul_6 = mul_7 = None
        cat_3 = torch.cat([getitem_22, add_5], dim=2)
        getitem_22 = add_5 = None
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
        reshape_6 = stack_3.reshape((1, 6, 196, 64))
        stack_3 = None
        mul_9 = reshape_6 * sin_emb_3
        reshape_6 = sin_emb_3 = None
        add_6 = mul_8 + mul_9
        mul_8 = mul_9 = None
        cat_4 = torch.cat([getitem_28, add_6], dim=2)
        getitem_28 = add_6 = None
        k_3 = cat_4.type_as(v_1)
        cat_4 = None
        x_19 = torch._C._nn.scaled_dot_product_attention(
            q_3, k_3, v_1, attn_mask=None, dropout_p=0.0
        )
        q_3 = k_3 = v_1 = None
        transpose_2 = x_19.transpose(1, 2)
        x_19 = None
        x_20 = transpose_2.reshape(1, 197, 384)
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
            (384,),
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
        x_31 = torch.nn.functional.layer_norm(
            x_30,
            (384,),
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
        reshape_8 = qkv_4.reshape(1, 197, 3, 6, -1)
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
        reshape_9 = stack_4.reshape((1, 6, 196, 64))
        stack_4 = None
        mul_13 = reshape_9 * sin_emb_4
        reshape_9 = sin_emb_4 = None
        add_9 = mul_12 + mul_13
        mul_12 = mul_13 = None
        cat_5 = torch.cat([getitem_37, add_9], dim=2)
        getitem_37 = add_9 = None
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
        reshape_10 = stack_5.reshape((1, 6, 196, 64))
        stack_5 = None
        mul_15 = reshape_10 * sin_emb_5
        reshape_10 = sin_emb_5 = None
        add_10 = mul_14 + mul_15
        mul_14 = mul_15 = None
        cat_6 = torch.cat([getitem_43, add_10], dim=2)
        getitem_43 = add_10 = None
        k_5 = cat_6.type_as(v_2)
        cat_6 = None
        x_32 = torch._C._nn.scaled_dot_product_attention(
            q_5, k_5, v_2, attn_mask=None, dropout_p=0.0
        )
        q_5 = k_5 = v_2 = None
        transpose_3 = x_32.transpose(1, 2)
        x_32 = None
        x_33 = transpose_3.reshape(1, 197, 384)
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
            (384,),
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
        x_44 = torch.nn.functional.layer_norm(
            x_43,
            (384,),
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
        reshape_12 = qkv_6.reshape(1, 197, 3, 6, -1)
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
        reshape_13 = stack_6.reshape((1, 6, 196, 64))
        stack_6 = None
        mul_19 = reshape_13 * sin_emb_6
        reshape_13 = sin_emb_6 = None
        add_13 = mul_18 + mul_19
        mul_18 = mul_19 = None
        cat_7 = torch.cat([getitem_52, add_13], dim=2)
        getitem_52 = add_13 = None
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
        reshape_14 = stack_7.reshape((1, 6, 196, 64))
        stack_7 = None
        mul_21 = reshape_14 * sin_emb_7
        reshape_14 = sin_emb_7 = None
        add_14 = mul_20 + mul_21
        mul_20 = mul_21 = None
        cat_8 = torch.cat([getitem_58, add_14], dim=2)
        getitem_58 = add_14 = None
        k_7 = cat_8.type_as(v_3)
        cat_8 = None
        x_45 = torch._C._nn.scaled_dot_product_attention(
            q_7, k_7, v_3, attn_mask=None, dropout_p=0.0
        )
        q_7 = k_7 = v_3 = None
        transpose_4 = x_45.transpose(1, 2)
        x_45 = None
        x_46 = transpose_4.reshape(1, 197, 384)
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
            (384,),
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
        x_57 = torch.nn.functional.layer_norm(
            x_56,
            (384,),
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
        reshape_16 = qkv_8.reshape(1, 197, 3, 6, -1)
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
        reshape_17 = stack_8.reshape((1, 6, 196, 64))
        stack_8 = None
        mul_25 = reshape_17 * sin_emb_8
        reshape_17 = sin_emb_8 = None
        add_17 = mul_24 + mul_25
        mul_24 = mul_25 = None
        cat_9 = torch.cat([getitem_67, add_17], dim=2)
        getitem_67 = add_17 = None
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
        reshape_18 = stack_9.reshape((1, 6, 196, 64))
        stack_9 = None
        mul_27 = reshape_18 * sin_emb_9
        reshape_18 = sin_emb_9 = None
        add_18 = mul_26 + mul_27
        mul_26 = mul_27 = None
        cat_10 = torch.cat([getitem_73, add_18], dim=2)
        getitem_73 = add_18 = None
        k_9 = cat_10.type_as(v_4)
        cat_10 = None
        x_58 = torch._C._nn.scaled_dot_product_attention(
            q_9, k_9, v_4, attn_mask=None, dropout_p=0.0
        )
        q_9 = k_9 = v_4 = None
        transpose_5 = x_58.transpose(1, 2)
        x_58 = None
        x_59 = transpose_5.reshape(1, 197, 384)
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
            (384,),
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
        x_70 = torch.nn.functional.layer_norm(
            x_69,
            (384,),
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
        reshape_20 = qkv_10.reshape(1, 197, 3, 6, -1)
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
        reshape_21 = stack_10.reshape((1, 6, 196, 64))
        stack_10 = None
        mul_31 = reshape_21 * sin_emb_10
        reshape_21 = sin_emb_10 = None
        add_21 = mul_30 + mul_31
        mul_30 = mul_31 = None
        cat_11 = torch.cat([getitem_82, add_21], dim=2)
        getitem_82 = add_21 = None
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
        reshape_22 = stack_11.reshape((1, 6, 196, 64))
        stack_11 = None
        mul_33 = reshape_22 * sin_emb_11
        reshape_22 = sin_emb_11 = None
        add_22 = mul_32 + mul_33
        mul_32 = mul_33 = None
        cat_12 = torch.cat([getitem_88, add_22], dim=2)
        getitem_88 = add_22 = None
        k_11 = cat_12.type_as(v_5)
        cat_12 = None
        x_71 = torch._C._nn.scaled_dot_product_attention(
            q_11, k_11, v_5, attn_mask=None, dropout_p=0.0
        )
        q_11 = k_11 = v_5 = None
        transpose_6 = x_71.transpose(1, 2)
        x_71 = None
        x_72 = transpose_6.reshape(1, 197, 384)
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
            (384,),
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
        x_83 = torch.nn.functional.layer_norm(
            x_82,
            (384,),
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
        reshape_24 = qkv_12.reshape(1, 197, 3, 6, -1)
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
        reshape_25 = stack_12.reshape((1, 6, 196, 64))
        stack_12 = None
        mul_37 = reshape_25 * sin_emb_12
        reshape_25 = sin_emb_12 = None
        add_25 = mul_36 + mul_37
        mul_36 = mul_37 = None
        cat_13 = torch.cat([getitem_97, add_25], dim=2)
        getitem_97 = add_25 = None
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
        reshape_26 = stack_13.reshape((1, 6, 196, 64))
        stack_13 = None
        mul_39 = reshape_26 * sin_emb_13
        reshape_26 = sin_emb_13 = None
        add_26 = mul_38 + mul_39
        mul_38 = mul_39 = None
        cat_14 = torch.cat([getitem_103, add_26], dim=2)
        getitem_103 = add_26 = None
        k_13 = cat_14.type_as(v_6)
        cat_14 = None
        x_84 = torch._C._nn.scaled_dot_product_attention(
            q_13, k_13, v_6, attn_mask=None, dropout_p=0.0
        )
        q_13 = k_13 = v_6 = None
        transpose_7 = x_84.transpose(1, 2)
        x_84 = None
        x_85 = transpose_7.reshape(1, 197, 384)
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
            (384,),
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
        x_96 = torch.nn.functional.layer_norm(
            x_95,
            (384,),
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
        reshape_28 = qkv_14.reshape(1, 197, 3, 6, -1)
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
        reshape_29 = stack_14.reshape((1, 6, 196, 64))
        stack_14 = None
        mul_43 = reshape_29 * sin_emb_14
        reshape_29 = sin_emb_14 = None
        add_29 = mul_42 + mul_43
        mul_42 = mul_43 = None
        cat_15 = torch.cat([getitem_112, add_29], dim=2)
        getitem_112 = add_29 = None
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
        reshape_30 = stack_15.reshape((1, 6, 196, 64))
        stack_15 = None
        mul_45 = reshape_30 * sin_emb_15
        reshape_30 = sin_emb_15 = None
        add_30 = mul_44 + mul_45
        mul_44 = mul_45 = None
        cat_16 = torch.cat([getitem_118, add_30], dim=2)
        getitem_118 = add_30 = None
        k_15 = cat_16.type_as(v_7)
        cat_16 = None
        x_97 = torch._C._nn.scaled_dot_product_attention(
            q_15, k_15, v_7, attn_mask=None, dropout_p=0.0
        )
        q_15 = k_15 = v_7 = None
        transpose_8 = x_97.transpose(1, 2)
        x_97 = None
        x_98 = transpose_8.reshape(1, 197, 384)
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
            (384,),
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
        x_109 = torch.nn.functional.layer_norm(
            x_108,
            (384,),
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
        reshape_32 = qkv_16.reshape(1, 197, 3, 6, -1)
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
        reshape_33 = stack_16.reshape((1, 6, 196, 64))
        stack_16 = None
        mul_49 = reshape_33 * sin_emb_16
        reshape_33 = sin_emb_16 = None
        add_33 = mul_48 + mul_49
        mul_48 = mul_49 = None
        cat_17 = torch.cat([getitem_127, add_33], dim=2)
        getitem_127 = add_33 = None
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
        reshape_34 = stack_17.reshape((1, 6, 196, 64))
        stack_17 = None
        mul_51 = reshape_34 * sin_emb_17
        reshape_34 = sin_emb_17 = None
        add_34 = mul_50 + mul_51
        mul_50 = mul_51 = None
        cat_18 = torch.cat([getitem_133, add_34], dim=2)
        getitem_133 = add_34 = None
        k_17 = cat_18.type_as(v_8)
        cat_18 = None
        x_110 = torch._C._nn.scaled_dot_product_attention(
            q_17, k_17, v_8, attn_mask=None, dropout_p=0.0
        )
        q_17 = k_17 = v_8 = None
        transpose_9 = x_110.transpose(1, 2)
        x_110 = None
        x_111 = transpose_9.reshape(1, 197, 384)
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
            (384,),
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
        x_122 = torch.nn.functional.layer_norm(
            x_121,
            (384,),
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
        reshape_36 = qkv_18.reshape(1, 197, 3, 6, -1)
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
        reshape_37 = stack_18.reshape((1, 6, 196, 64))
        stack_18 = None
        mul_55 = reshape_37 * sin_emb_18
        reshape_37 = sin_emb_18 = None
        add_37 = mul_54 + mul_55
        mul_54 = mul_55 = None
        cat_19 = torch.cat([getitem_142, add_37], dim=2)
        getitem_142 = add_37 = None
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
        reshape_38 = stack_19.reshape((1, 6, 196, 64))
        stack_19 = None
        mul_57 = reshape_38 * sin_emb_19
        reshape_38 = sin_emb_19 = None
        add_38 = mul_56 + mul_57
        mul_56 = mul_57 = None
        cat_20 = torch.cat([getitem_148, add_38], dim=2)
        getitem_148 = add_38 = None
        k_19 = cat_20.type_as(v_9)
        cat_20 = None
        x_123 = torch._C._nn.scaled_dot_product_attention(
            q_19, k_19, v_9, attn_mask=None, dropout_p=0.0
        )
        q_19 = k_19 = v_9 = None
        transpose_10 = x_123.transpose(1, 2)
        x_123 = None
        x_124 = transpose_10.reshape(1, 197, 384)
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
            (384,),
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
        x_135 = torch.nn.functional.layer_norm(
            x_134,
            (384,),
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
        reshape_40 = qkv_20.reshape(1, 197, 3, 6, -1)
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
        reshape_41 = stack_20.reshape((1, 6, 196, 64))
        stack_20 = None
        mul_61 = reshape_41 * sin_emb_20
        reshape_41 = sin_emb_20 = None
        add_41 = mul_60 + mul_61
        mul_60 = mul_61 = None
        cat_21 = torch.cat([getitem_157, add_41], dim=2)
        getitem_157 = add_41 = None
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
        reshape_42 = stack_21.reshape((1, 6, 196, 64))
        stack_21 = None
        mul_63 = reshape_42 * sin_emb_21
        reshape_42 = sin_emb_21 = None
        add_42 = mul_62 + mul_63
        mul_62 = mul_63 = None
        cat_22 = torch.cat([getitem_163, add_42], dim=2)
        getitem_163 = add_42 = None
        k_21 = cat_22.type_as(v_10)
        cat_22 = None
        x_136 = torch._C._nn.scaled_dot_product_attention(
            q_21, k_21, v_10, attn_mask=None, dropout_p=0.0
        )
        q_21 = k_21 = v_10 = None
        transpose_11 = x_136.transpose(1, 2)
        x_136 = None
        x_137 = transpose_11.reshape(1, 197, 384)
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
            (384,),
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
        x_148 = torch.nn.functional.layer_norm(
            x_147,
            (384,),
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
        reshape_44 = qkv_22.reshape(1, 197, 3, 6, -1)
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
        reshape_45 = stack_22.reshape((1, 6, 196, 64))
        stack_22 = None
        mul_67 = reshape_45 * sin_emb_22
        reshape_45 = sin_emb_22 = None
        add_45 = mul_66 + mul_67
        mul_66 = mul_67 = None
        cat_23 = torch.cat([getitem_172, add_45], dim=2)
        getitem_172 = add_45 = None
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
        l_self_modules_rope_buffers_pos_embed_ = None
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
        reshape_46 = stack_23.reshape((1, 6, 196, 64))
        stack_23 = None
        mul_69 = reshape_46 * sin_emb_23
        reshape_46 = sin_emb_23 = None
        add_46 = mul_68 + mul_69
        mul_68 = mul_69 = None
        cat_24 = torch.cat([getitem_178, add_46], dim=2)
        getitem_178 = add_46 = None
        k_23 = cat_24.type_as(v_11)
        cat_24 = None
        x_149 = torch._C._nn.scaled_dot_product_attention(
            q_23, k_23, v_11, attn_mask=None, dropout_p=0.0
        )
        q_23 = k_23 = v_11 = None
        transpose_12 = x_149.transpose(1, 2)
        x_149 = None
        x_150 = transpose_12.reshape(1, 197, 384)
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
            (384,),
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
        x_161 = torch.nn.functional.layer_norm(
            x_160,
            (384,),
            l_self_modules_norm_parameters_weight_,
            l_self_modules_norm_parameters_bias_,
            1e-06,
        )
        x_160 = (
            l_self_modules_norm_parameters_weight_
        ) = l_self_modules_norm_parameters_bias_ = None
        x_162 = x_161[(slice(None, None, None), 0)]
        x_161 = None
        x_163 = torch.nn.functional.dropout(x_162, 0.0, False, False)
        x_162 = None
        x_164 = torch._C._nn.linear(
            x_163,
            l_self_modules_head_parameters_weight_,
            l_self_modules_head_parameters_bias_,
        )
        x_163 = (
            l_self_modules_head_parameters_weight_
        ) = l_self_modules_head_parameters_bias_ = None
        return (x_164,)
