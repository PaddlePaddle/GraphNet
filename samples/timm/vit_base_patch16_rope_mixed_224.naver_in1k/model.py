import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_x_: torch.Tensor,
        L_self_modules_patch_embed_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_patch_embed_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
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
        x_2 = torch.cat([expand, x_1], dim=1)
        expand = x_1 = None
        x_3 = torch.nn.functional.dropout(x_2, 0.0, False, False)
        x_2 = None
        getitem_6 = rot_pos_embed[0]
        x_4 = torch.nn.functional.layer_norm(
            x_3,
            (768,),
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
        reshape = qkv.reshape(1, 197, 3, 12, -1)
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
        reshape_1 = stack.reshape((1, 12, 196, 64))
        stack = None
        mul_1 = reshape_1 * sin_emb_1
        reshape_1 = sin_emb_1 = None
        add_1 = mul + mul_1
        mul = mul_1 = None
        cat_2 = torch.cat([getitem_10, add_1], dim=2)
        getitem_10 = add_1 = None
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
        reshape_2 = stack_1.reshape((1, 12, 196, 64))
        stack_1 = None
        mul_3 = reshape_2 * sin_emb_2
        reshape_2 = sin_emb_2 = None
        add_2 = mul_2 + mul_3
        mul_2 = mul_3 = None
        cat_3 = torch.cat([getitem_16, add_2], dim=2)
        getitem_16 = add_2 = None
        k_1 = cat_3.type_as(v)
        cat_3 = None
        x_5 = torch._C._nn.scaled_dot_product_attention(
            q_1, k_1, v, attn_mask=None, dropout_p=0.0
        )
        q_1 = k_1 = v = None
        transpose_1 = x_5.transpose(1, 2)
        x_5 = None
        x_6 = transpose_1.reshape(1, 197, 768)
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
            (768,),
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
        getitem_22 = rot_pos_embed[1]
        x_17 = torch.nn.functional.layer_norm(
            x_16,
            (768,),
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
        reshape_4 = qkv_2.reshape(1, 197, 3, 12, -1)
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
        reshape_5 = stack_2.reshape((1, 12, 196, 64))
        stack_2 = None
        mul_7 = reshape_5 * sin_emb_3
        reshape_5 = sin_emb_3 = None
        add_5 = mul_6 + mul_7
        mul_6 = mul_7 = None
        cat_4 = torch.cat([getitem_26, add_5], dim=2)
        getitem_26 = add_5 = None
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
        reshape_6 = stack_3.reshape((1, 12, 196, 64))
        stack_3 = None
        mul_9 = reshape_6 * sin_emb_4
        reshape_6 = sin_emb_4 = None
        add_6 = mul_8 + mul_9
        mul_8 = mul_9 = None
        cat_5 = torch.cat([getitem_32, add_6], dim=2)
        getitem_32 = add_6 = None
        k_3 = cat_5.type_as(v_1)
        cat_5 = None
        x_18 = torch._C._nn.scaled_dot_product_attention(
            q_3, k_3, v_1, attn_mask=None, dropout_p=0.0
        )
        q_3 = k_3 = v_1 = None
        transpose_2 = x_18.transpose(1, 2)
        x_18 = None
        x_19 = transpose_2.reshape(1, 197, 768)
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
            (768,),
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
        getitem_38 = rot_pos_embed[2]
        x_30 = torch.nn.functional.layer_norm(
            x_29,
            (768,),
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
        reshape_8 = qkv_4.reshape(1, 197, 3, 12, -1)
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
        reshape_9 = stack_4.reshape((1, 12, 196, 64))
        stack_4 = None
        mul_13 = reshape_9 * sin_emb_5
        reshape_9 = sin_emb_5 = None
        add_9 = mul_12 + mul_13
        mul_12 = mul_13 = None
        cat_6 = torch.cat([getitem_42, add_9], dim=2)
        getitem_42 = add_9 = None
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
        reshape_10 = stack_5.reshape((1, 12, 196, 64))
        stack_5 = None
        mul_15 = reshape_10 * sin_emb_6
        reshape_10 = sin_emb_6 = None
        add_10 = mul_14 + mul_15
        mul_14 = mul_15 = None
        cat_7 = torch.cat([getitem_48, add_10], dim=2)
        getitem_48 = add_10 = None
        k_5 = cat_7.type_as(v_2)
        cat_7 = None
        x_31 = torch._C._nn.scaled_dot_product_attention(
            q_5, k_5, v_2, attn_mask=None, dropout_p=0.0
        )
        q_5 = k_5 = v_2 = None
        transpose_3 = x_31.transpose(1, 2)
        x_31 = None
        x_32 = transpose_3.reshape(1, 197, 768)
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
            (768,),
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
        getitem_54 = rot_pos_embed[3]
        x_43 = torch.nn.functional.layer_norm(
            x_42,
            (768,),
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
        reshape_12 = qkv_6.reshape(1, 197, 3, 12, -1)
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
        reshape_13 = stack_6.reshape((1, 12, 196, 64))
        stack_6 = None
        mul_19 = reshape_13 * sin_emb_7
        reshape_13 = sin_emb_7 = None
        add_13 = mul_18 + mul_19
        mul_18 = mul_19 = None
        cat_8 = torch.cat([getitem_58, add_13], dim=2)
        getitem_58 = add_13 = None
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
        reshape_14 = stack_7.reshape((1, 12, 196, 64))
        stack_7 = None
        mul_21 = reshape_14 * sin_emb_8
        reshape_14 = sin_emb_8 = None
        add_14 = mul_20 + mul_21
        mul_20 = mul_21 = None
        cat_9 = torch.cat([getitem_64, add_14], dim=2)
        getitem_64 = add_14 = None
        k_7 = cat_9.type_as(v_3)
        cat_9 = None
        x_44 = torch._C._nn.scaled_dot_product_attention(
            q_7, k_7, v_3, attn_mask=None, dropout_p=0.0
        )
        q_7 = k_7 = v_3 = None
        transpose_4 = x_44.transpose(1, 2)
        x_44 = None
        x_45 = transpose_4.reshape(1, 197, 768)
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
            (768,),
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
        getitem_70 = rot_pos_embed[4]
        x_56 = torch.nn.functional.layer_norm(
            x_55,
            (768,),
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
        reshape_16 = qkv_8.reshape(1, 197, 3, 12, -1)
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
        reshape_17 = stack_8.reshape((1, 12, 196, 64))
        stack_8 = None
        mul_25 = reshape_17 * sin_emb_9
        reshape_17 = sin_emb_9 = None
        add_17 = mul_24 + mul_25
        mul_24 = mul_25 = None
        cat_10 = torch.cat([getitem_74, add_17], dim=2)
        getitem_74 = add_17 = None
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
        reshape_18 = stack_9.reshape((1, 12, 196, 64))
        stack_9 = None
        mul_27 = reshape_18 * sin_emb_10
        reshape_18 = sin_emb_10 = None
        add_18 = mul_26 + mul_27
        mul_26 = mul_27 = None
        cat_11 = torch.cat([getitem_80, add_18], dim=2)
        getitem_80 = add_18 = None
        k_9 = cat_11.type_as(v_4)
        cat_11 = None
        x_57 = torch._C._nn.scaled_dot_product_attention(
            q_9, k_9, v_4, attn_mask=None, dropout_p=0.0
        )
        q_9 = k_9 = v_4 = None
        transpose_5 = x_57.transpose(1, 2)
        x_57 = None
        x_58 = transpose_5.reshape(1, 197, 768)
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
            (768,),
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
        getitem_86 = rot_pos_embed[5]
        x_69 = torch.nn.functional.layer_norm(
            x_68,
            (768,),
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
        reshape_20 = qkv_10.reshape(1, 197, 3, 12, -1)
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
        reshape_21 = stack_10.reshape((1, 12, 196, 64))
        stack_10 = None
        mul_31 = reshape_21 * sin_emb_11
        reshape_21 = sin_emb_11 = None
        add_21 = mul_30 + mul_31
        mul_30 = mul_31 = None
        cat_12 = torch.cat([getitem_90, add_21], dim=2)
        getitem_90 = add_21 = None
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
        reshape_22 = stack_11.reshape((1, 12, 196, 64))
        stack_11 = None
        mul_33 = reshape_22 * sin_emb_12
        reshape_22 = sin_emb_12 = None
        add_22 = mul_32 + mul_33
        mul_32 = mul_33 = None
        cat_13 = torch.cat([getitem_96, add_22], dim=2)
        getitem_96 = add_22 = None
        k_11 = cat_13.type_as(v_5)
        cat_13 = None
        x_70 = torch._C._nn.scaled_dot_product_attention(
            q_11, k_11, v_5, attn_mask=None, dropout_p=0.0
        )
        q_11 = k_11 = v_5 = None
        transpose_6 = x_70.transpose(1, 2)
        x_70 = None
        x_71 = transpose_6.reshape(1, 197, 768)
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
            (768,),
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
        getitem_102 = rot_pos_embed[6]
        x_82 = torch.nn.functional.layer_norm(
            x_81,
            (768,),
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
        reshape_24 = qkv_12.reshape(1, 197, 3, 12, -1)
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
        reshape_25 = stack_12.reshape((1, 12, 196, 64))
        stack_12 = None
        mul_37 = reshape_25 * sin_emb_13
        reshape_25 = sin_emb_13 = None
        add_25 = mul_36 + mul_37
        mul_36 = mul_37 = None
        cat_14 = torch.cat([getitem_106, add_25], dim=2)
        getitem_106 = add_25 = None
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
        reshape_26 = stack_13.reshape((1, 12, 196, 64))
        stack_13 = None
        mul_39 = reshape_26 * sin_emb_14
        reshape_26 = sin_emb_14 = None
        add_26 = mul_38 + mul_39
        mul_38 = mul_39 = None
        cat_15 = torch.cat([getitem_112, add_26], dim=2)
        getitem_112 = add_26 = None
        k_13 = cat_15.type_as(v_6)
        cat_15 = None
        x_83 = torch._C._nn.scaled_dot_product_attention(
            q_13, k_13, v_6, attn_mask=None, dropout_p=0.0
        )
        q_13 = k_13 = v_6 = None
        transpose_7 = x_83.transpose(1, 2)
        x_83 = None
        x_84 = transpose_7.reshape(1, 197, 768)
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
            (768,),
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
        getitem_118 = rot_pos_embed[7]
        x_95 = torch.nn.functional.layer_norm(
            x_94,
            (768,),
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
        reshape_28 = qkv_14.reshape(1, 197, 3, 12, -1)
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
        reshape_29 = stack_14.reshape((1, 12, 196, 64))
        stack_14 = None
        mul_43 = reshape_29 * sin_emb_15
        reshape_29 = sin_emb_15 = None
        add_29 = mul_42 + mul_43
        mul_42 = mul_43 = None
        cat_16 = torch.cat([getitem_122, add_29], dim=2)
        getitem_122 = add_29 = None
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
        reshape_30 = stack_15.reshape((1, 12, 196, 64))
        stack_15 = None
        mul_45 = reshape_30 * sin_emb_16
        reshape_30 = sin_emb_16 = None
        add_30 = mul_44 + mul_45
        mul_44 = mul_45 = None
        cat_17 = torch.cat([getitem_128, add_30], dim=2)
        getitem_128 = add_30 = None
        k_15 = cat_17.type_as(v_7)
        cat_17 = None
        x_96 = torch._C._nn.scaled_dot_product_attention(
            q_15, k_15, v_7, attn_mask=None, dropout_p=0.0
        )
        q_15 = k_15 = v_7 = None
        transpose_8 = x_96.transpose(1, 2)
        x_96 = None
        x_97 = transpose_8.reshape(1, 197, 768)
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
            (768,),
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
        getitem_134 = rot_pos_embed[8]
        x_108 = torch.nn.functional.layer_norm(
            x_107,
            (768,),
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
        reshape_32 = qkv_16.reshape(1, 197, 3, 12, -1)
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
        reshape_33 = stack_16.reshape((1, 12, 196, 64))
        stack_16 = None
        mul_49 = reshape_33 * sin_emb_17
        reshape_33 = sin_emb_17 = None
        add_33 = mul_48 + mul_49
        mul_48 = mul_49 = None
        cat_18 = torch.cat([getitem_138, add_33], dim=2)
        getitem_138 = add_33 = None
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
        reshape_34 = stack_17.reshape((1, 12, 196, 64))
        stack_17 = None
        mul_51 = reshape_34 * sin_emb_18
        reshape_34 = sin_emb_18 = None
        add_34 = mul_50 + mul_51
        mul_50 = mul_51 = None
        cat_19 = torch.cat([getitem_144, add_34], dim=2)
        getitem_144 = add_34 = None
        k_17 = cat_19.type_as(v_8)
        cat_19 = None
        x_109 = torch._C._nn.scaled_dot_product_attention(
            q_17, k_17, v_8, attn_mask=None, dropout_p=0.0
        )
        q_17 = k_17 = v_8 = None
        transpose_9 = x_109.transpose(1, 2)
        x_109 = None
        x_110 = transpose_9.reshape(1, 197, 768)
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
            (768,),
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
        getitem_150 = rot_pos_embed[9]
        x_121 = torch.nn.functional.layer_norm(
            x_120,
            (768,),
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
        reshape_36 = qkv_18.reshape(1, 197, 3, 12, -1)
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
        reshape_37 = stack_18.reshape((1, 12, 196, 64))
        stack_18 = None
        mul_55 = reshape_37 * sin_emb_19
        reshape_37 = sin_emb_19 = None
        add_37 = mul_54 + mul_55
        mul_54 = mul_55 = None
        cat_20 = torch.cat([getitem_154, add_37], dim=2)
        getitem_154 = add_37 = None
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
        reshape_38 = stack_19.reshape((1, 12, 196, 64))
        stack_19 = None
        mul_57 = reshape_38 * sin_emb_20
        reshape_38 = sin_emb_20 = None
        add_38 = mul_56 + mul_57
        mul_56 = mul_57 = None
        cat_21 = torch.cat([getitem_160, add_38], dim=2)
        getitem_160 = add_38 = None
        k_19 = cat_21.type_as(v_9)
        cat_21 = None
        x_122 = torch._C._nn.scaled_dot_product_attention(
            q_19, k_19, v_9, attn_mask=None, dropout_p=0.0
        )
        q_19 = k_19 = v_9 = None
        transpose_10 = x_122.transpose(1, 2)
        x_122 = None
        x_123 = transpose_10.reshape(1, 197, 768)
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
            (768,),
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
        getitem_166 = rot_pos_embed[10]
        x_134 = torch.nn.functional.layer_norm(
            x_133,
            (768,),
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
        reshape_40 = qkv_20.reshape(1, 197, 3, 12, -1)
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
        reshape_41 = stack_20.reshape((1, 12, 196, 64))
        stack_20 = None
        mul_61 = reshape_41 * sin_emb_21
        reshape_41 = sin_emb_21 = None
        add_41 = mul_60 + mul_61
        mul_60 = mul_61 = None
        cat_22 = torch.cat([getitem_170, add_41], dim=2)
        getitem_170 = add_41 = None
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
        reshape_42 = stack_21.reshape((1, 12, 196, 64))
        stack_21 = None
        mul_63 = reshape_42 * sin_emb_22
        reshape_42 = sin_emb_22 = None
        add_42 = mul_62 + mul_63
        mul_62 = mul_63 = None
        cat_23 = torch.cat([getitem_176, add_42], dim=2)
        getitem_176 = add_42 = None
        k_21 = cat_23.type_as(v_10)
        cat_23 = None
        x_135 = torch._C._nn.scaled_dot_product_attention(
            q_21, k_21, v_10, attn_mask=None, dropout_p=0.0
        )
        q_21 = k_21 = v_10 = None
        transpose_11 = x_135.transpose(1, 2)
        x_135 = None
        x_136 = transpose_11.reshape(1, 197, 768)
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
            (768,),
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
        getitem_182 = rot_pos_embed[11]
        rot_pos_embed = None
        x_147 = torch.nn.functional.layer_norm(
            x_146,
            (768,),
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
        reshape_44 = qkv_22.reshape(1, 197, 3, 12, -1)
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
        reshape_45 = stack_22.reshape((1, 12, 196, 64))
        stack_22 = None
        mul_67 = reshape_45 * sin_emb_23
        reshape_45 = sin_emb_23 = None
        add_45 = mul_66 + mul_67
        mul_66 = mul_67 = None
        cat_24 = torch.cat([getitem_186, add_45], dim=2)
        getitem_186 = add_45 = None
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
        reshape_46 = stack_23.reshape((1, 12, 196, 64))
        stack_23 = None
        mul_69 = reshape_46 * sin_emb_24
        reshape_46 = sin_emb_24 = None
        add_46 = mul_68 + mul_69
        mul_68 = mul_69 = None
        cat_25 = torch.cat([getitem_192, add_46], dim=2)
        getitem_192 = add_46 = None
        k_23 = cat_25.type_as(v_11)
        cat_25 = None
        x_148 = torch._C._nn.scaled_dot_product_attention(
            q_23, k_23, v_11, attn_mask=None, dropout_p=0.0
        )
        q_23 = k_23 = v_11 = None
        transpose_12 = x_148.transpose(1, 2)
        x_148 = None
        x_149 = transpose_12.reshape(1, 197, 768)
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
            (768,),
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
            (768,),
            l_self_modules_norm_parameters_weight_,
            l_self_modules_norm_parameters_bias_,
            1e-06,
        )
        x_159 = (
            l_self_modules_norm_parameters_weight_
        ) = l_self_modules_norm_parameters_bias_ = None
        x_161 = x_160[(slice(None, None, None), 0)]
        x_160 = None
        x_162 = torch.nn.functional.dropout(x_161, 0.0, False, False)
        x_161 = None
        x_163 = torch._C._nn.linear(
            x_162,
            l_self_modules_head_parameters_weight_,
            l_self_modules_head_parameters_bias_,
        )
        x_162 = (
            l_self_modules_head_parameters_weight_
        ) = l_self_modules_head_parameters_bias_ = None
        return (x_163,)
