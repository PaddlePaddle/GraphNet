import torch

from torch import device


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
        L_self_modules_blocks_modules_0_modules_attn_modules_qk_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_attn_modules_pos_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_attn_modules_pos_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_attn_parameters_gating_param_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_attn_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_1_modules_attn_modules_qk_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_attn_modules_pos_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_attn_modules_pos_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_attn_parameters_gating_param_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_attn_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_2_modules_attn_modules_qk_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_attn_modules_pos_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_attn_modules_pos_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_attn_parameters_gating_param_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_attn_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_3_modules_attn_modules_qk_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_attn_modules_pos_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_attn_modules_pos_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_attn_parameters_gating_param_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_attn_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_4_modules_attn_modules_qk_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_attn_modules_pos_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_attn_modules_pos_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_attn_parameters_gating_param_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_attn_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_5_modules_attn_modules_qk_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_attn_modules_pos_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_attn_modules_pos_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_attn_parameters_gating_param_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_attn_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_6_modules_attn_modules_qk_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_attn_modules_pos_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_attn_modules_pos_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_attn_parameters_gating_param_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_attn_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_7_modules_attn_modules_qk_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_attn_modules_pos_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_attn_modules_pos_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_attn_parameters_gating_param_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_attn_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_8_modules_attn_modules_qk_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_attn_modules_pos_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_attn_modules_pos_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_attn_parameters_gating_param_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_attn_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_9_modules_attn_modules_qk_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_attn_modules_pos_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_attn_modules_pos_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_attn_parameters_gating_param_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_attn_modules_v_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_parameters_cls_token_ = L_self_parameters_cls_token_
        l_self_modules_blocks_modules_0_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_0_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_0_modules_attn_modules_qk_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_attn_modules_qk_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_attn_modules_pos_proj_parameters_weight_ = L_self_modules_blocks_modules_0_modules_attn_modules_pos_proj_parameters_weight_
        l_self_modules_blocks_modules_0_modules_attn_modules_pos_proj_parameters_bias_ = L_self_modules_blocks_modules_0_modules_attn_modules_pos_proj_parameters_bias_
        l_self_modules_blocks_modules_0_modules_attn_parameters_gating_param_ = (
            L_self_modules_blocks_modules_0_modules_attn_parameters_gating_param_
        )
        l_self_modules_blocks_modules_0_modules_attn_modules_v_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_attn_modules_v_parameters_weight_
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
        l_self_modules_blocks_modules_1_modules_attn_modules_qk_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_attn_modules_qk_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_attn_modules_pos_proj_parameters_weight_ = L_self_modules_blocks_modules_1_modules_attn_modules_pos_proj_parameters_weight_
        l_self_modules_blocks_modules_1_modules_attn_modules_pos_proj_parameters_bias_ = L_self_modules_blocks_modules_1_modules_attn_modules_pos_proj_parameters_bias_
        l_self_modules_blocks_modules_1_modules_attn_parameters_gating_param_ = (
            L_self_modules_blocks_modules_1_modules_attn_parameters_gating_param_
        )
        l_self_modules_blocks_modules_1_modules_attn_modules_v_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_attn_modules_v_parameters_weight_
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
        l_self_modules_blocks_modules_2_modules_attn_modules_qk_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_attn_modules_qk_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_attn_modules_pos_proj_parameters_weight_ = L_self_modules_blocks_modules_2_modules_attn_modules_pos_proj_parameters_weight_
        l_self_modules_blocks_modules_2_modules_attn_modules_pos_proj_parameters_bias_ = L_self_modules_blocks_modules_2_modules_attn_modules_pos_proj_parameters_bias_
        l_self_modules_blocks_modules_2_modules_attn_parameters_gating_param_ = (
            L_self_modules_blocks_modules_2_modules_attn_parameters_gating_param_
        )
        l_self_modules_blocks_modules_2_modules_attn_modules_v_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_attn_modules_v_parameters_weight_
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
        l_self_modules_blocks_modules_3_modules_attn_modules_qk_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_attn_modules_qk_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_attn_modules_pos_proj_parameters_weight_ = L_self_modules_blocks_modules_3_modules_attn_modules_pos_proj_parameters_weight_
        l_self_modules_blocks_modules_3_modules_attn_modules_pos_proj_parameters_bias_ = L_self_modules_blocks_modules_3_modules_attn_modules_pos_proj_parameters_bias_
        l_self_modules_blocks_modules_3_modules_attn_parameters_gating_param_ = (
            L_self_modules_blocks_modules_3_modules_attn_parameters_gating_param_
        )
        l_self_modules_blocks_modules_3_modules_attn_modules_v_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_attn_modules_v_parameters_weight_
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
        l_self_modules_blocks_modules_4_modules_attn_modules_qk_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_attn_modules_qk_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_attn_modules_pos_proj_parameters_weight_ = L_self_modules_blocks_modules_4_modules_attn_modules_pos_proj_parameters_weight_
        l_self_modules_blocks_modules_4_modules_attn_modules_pos_proj_parameters_bias_ = L_self_modules_blocks_modules_4_modules_attn_modules_pos_proj_parameters_bias_
        l_self_modules_blocks_modules_4_modules_attn_parameters_gating_param_ = (
            L_self_modules_blocks_modules_4_modules_attn_parameters_gating_param_
        )
        l_self_modules_blocks_modules_4_modules_attn_modules_v_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_attn_modules_v_parameters_weight_
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
        l_self_modules_blocks_modules_5_modules_attn_modules_qk_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_attn_modules_qk_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_attn_modules_pos_proj_parameters_weight_ = L_self_modules_blocks_modules_5_modules_attn_modules_pos_proj_parameters_weight_
        l_self_modules_blocks_modules_5_modules_attn_modules_pos_proj_parameters_bias_ = L_self_modules_blocks_modules_5_modules_attn_modules_pos_proj_parameters_bias_
        l_self_modules_blocks_modules_5_modules_attn_parameters_gating_param_ = (
            L_self_modules_blocks_modules_5_modules_attn_parameters_gating_param_
        )
        l_self_modules_blocks_modules_5_modules_attn_modules_v_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_attn_modules_v_parameters_weight_
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
        l_self_modules_blocks_modules_6_modules_attn_modules_qk_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_attn_modules_qk_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_attn_modules_pos_proj_parameters_weight_ = L_self_modules_blocks_modules_6_modules_attn_modules_pos_proj_parameters_weight_
        l_self_modules_blocks_modules_6_modules_attn_modules_pos_proj_parameters_bias_ = L_self_modules_blocks_modules_6_modules_attn_modules_pos_proj_parameters_bias_
        l_self_modules_blocks_modules_6_modules_attn_parameters_gating_param_ = (
            L_self_modules_blocks_modules_6_modules_attn_parameters_gating_param_
        )
        l_self_modules_blocks_modules_6_modules_attn_modules_v_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_attn_modules_v_parameters_weight_
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
        l_self_modules_blocks_modules_7_modules_attn_modules_qk_parameters_weight_ = (
            L_self_modules_blocks_modules_7_modules_attn_modules_qk_parameters_weight_
        )
        l_self_modules_blocks_modules_7_modules_attn_modules_pos_proj_parameters_weight_ = L_self_modules_blocks_modules_7_modules_attn_modules_pos_proj_parameters_weight_
        l_self_modules_blocks_modules_7_modules_attn_modules_pos_proj_parameters_bias_ = L_self_modules_blocks_modules_7_modules_attn_modules_pos_proj_parameters_bias_
        l_self_modules_blocks_modules_7_modules_attn_parameters_gating_param_ = (
            L_self_modules_blocks_modules_7_modules_attn_parameters_gating_param_
        )
        l_self_modules_blocks_modules_7_modules_attn_modules_v_parameters_weight_ = (
            L_self_modules_blocks_modules_7_modules_attn_modules_v_parameters_weight_
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
        l_self_modules_blocks_modules_8_modules_attn_modules_qk_parameters_weight_ = (
            L_self_modules_blocks_modules_8_modules_attn_modules_qk_parameters_weight_
        )
        l_self_modules_blocks_modules_8_modules_attn_modules_pos_proj_parameters_weight_ = L_self_modules_blocks_modules_8_modules_attn_modules_pos_proj_parameters_weight_
        l_self_modules_blocks_modules_8_modules_attn_modules_pos_proj_parameters_bias_ = L_self_modules_blocks_modules_8_modules_attn_modules_pos_proj_parameters_bias_
        l_self_modules_blocks_modules_8_modules_attn_parameters_gating_param_ = (
            L_self_modules_blocks_modules_8_modules_attn_parameters_gating_param_
        )
        l_self_modules_blocks_modules_8_modules_attn_modules_v_parameters_weight_ = (
            L_self_modules_blocks_modules_8_modules_attn_modules_v_parameters_weight_
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
        l_self_modules_blocks_modules_9_modules_attn_modules_qk_parameters_weight_ = (
            L_self_modules_blocks_modules_9_modules_attn_modules_qk_parameters_weight_
        )
        l_self_modules_blocks_modules_9_modules_attn_modules_pos_proj_parameters_weight_ = L_self_modules_blocks_modules_9_modules_attn_modules_pos_proj_parameters_weight_
        l_self_modules_blocks_modules_9_modules_attn_modules_pos_proj_parameters_bias_ = L_self_modules_blocks_modules_9_modules_attn_modules_pos_proj_parameters_bias_
        l_self_modules_blocks_modules_9_modules_attn_parameters_gating_param_ = (
            L_self_modules_blocks_modules_9_modules_attn_parameters_gating_param_
        )
        l_self_modules_blocks_modules_9_modules_attn_modules_v_parameters_weight_ = (
            L_self_modules_blocks_modules_9_modules_attn_modules_v_parameters_weight_
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
        cls_tokens = l_self_parameters_cls_token_.expand(1, -1, -1)
        l_self_parameters_cls_token_ = None
        x_4 = torch.nn.functional.layer_norm(
            x_3,
            (432,),
            l_self_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_0_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_0_modules_norm1_parameters_bias_
        ) = None
        rel_indices = torch.zeros(1, 196, 196, 3)
        arange = torch.arange(14)
        view = arange.view(1, -1)
        arange = None
        arange_1 = torch.arange(14)
        view_1 = arange_1.view(-1, 1)
        arange_1 = None
        ind = view - view_1
        view = view_1 = None
        indx = ind.repeat(14, 14)
        repeat_interleave = ind.repeat_interleave(14, dim=0)
        ind = None
        indy = repeat_interleave.repeat_interleave(14, dim=1)
        repeat_interleave = None
        pow_1 = indx**2
        pow_2 = indy**2
        indd = pow_1 + pow_2
        pow_1 = pow_2 = None
        unsqueeze = indd.unsqueeze(0)
        indd = None
        rel_indices[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                2,
            )
        ] = unsqueeze
        setitem = rel_indices
        unsqueeze = setitem = None
        unsqueeze_1 = indy.unsqueeze(0)
        indy = None
        rel_indices[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                1,
            )
        ] = unsqueeze_1
        setitem_1 = rel_indices
        unsqueeze_1 = setitem_1 = None
        unsqueeze_2 = indx.unsqueeze(0)
        indx = None
        rel_indices[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                0,
            )
        ] = unsqueeze_2
        setitem_2 = rel_indices
        unsqueeze_2 = setitem_2 = None
        to = rel_indices.to(device(type="cpu"))
        rel_indices = None
        linear = torch._C._nn.linear(
            x_4,
            l_self_modules_blocks_modules_0_modules_attn_modules_qk_parameters_weight_,
            None,
        )
        l_self_modules_blocks_modules_0_modules_attn_modules_qk_parameters_weight_ = (
            None
        )
        reshape = linear.reshape(1, 196, 2, 9, 48)
        linear = None
        qk = reshape.permute(2, 0, 3, 1, 4)
        reshape = None
        q = qk[0]
        k = qk[1]
        qk = None
        pos_score = to.expand(1, -1, -1, -1)
        linear_1 = torch._C._nn.linear(
            pos_score,
            l_self_modules_blocks_modules_0_modules_attn_modules_pos_proj_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_attn_modules_pos_proj_parameters_bias_,
        )
        pos_score = l_self_modules_blocks_modules_0_modules_attn_modules_pos_proj_parameters_weight_ = l_self_modules_blocks_modules_0_modules_attn_modules_pos_proj_parameters_bias_ = (None)
        pos_score_1 = linear_1.permute(0, 3, 1, 2)
        linear_1 = None
        transpose_1 = k.transpose(-2, -1)
        k = None
        matmul = q @ transpose_1
        q = transpose_1 = None
        patch_score = matmul * 0.14433756729740643
        matmul = None
        patch_score_1 = patch_score.softmax(dim=-1)
        patch_score = None
        pos_score_2 = pos_score_1.softmax(dim=-1)
        pos_score_1 = None
        gating = (
            l_self_modules_blocks_modules_0_modules_attn_parameters_gating_param_.view(
                1, -1, 1, 1
            )
        )
        l_self_modules_blocks_modules_0_modules_attn_parameters_gating_param_ = None
        sigmoid = torch.sigmoid(gating)
        sub_1 = 1.0 - sigmoid
        sigmoid = None
        mul_1 = sub_1 * patch_score_1
        sub_1 = patch_score_1 = None
        sigmoid_1 = torch.sigmoid(gating)
        gating = None
        mul_2 = sigmoid_1 * pos_score_2
        sigmoid_1 = pos_score_2 = None
        attn = mul_1 + mul_2
        mul_1 = mul_2 = None
        sum_1 = attn.sum(dim=-1)
        unsqueeze_3 = sum_1.unsqueeze(-1)
        sum_1 = None
        attn /= unsqueeze_3
        attn_1 = attn
        attn = unsqueeze_3 = None
        attn_2 = torch.nn.functional.dropout(attn_1, 0.0, False, False)
        attn_1 = None
        linear_2 = torch._C._nn.linear(
            x_4,
            l_self_modules_blocks_modules_0_modules_attn_modules_v_parameters_weight_,
            None,
        )
        x_4 = (
            l_self_modules_blocks_modules_0_modules_attn_modules_v_parameters_weight_
        ) = None
        reshape_1 = linear_2.reshape(1, 196, 9, 48)
        linear_2 = None
        v = reshape_1.permute(0, 2, 1, 3)
        reshape_1 = None
        matmul_1 = attn_2 @ v
        attn_2 = v = None
        transpose_2 = matmul_1.transpose(1, 2)
        matmul_1 = None
        x_5 = transpose_2.reshape(1, 196, 432)
        transpose_2 = None
        x_6 = torch._C._nn.linear(
            x_5,
            l_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_5 = (
            l_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_7 = torch.nn.functional.dropout(x_6, 0.0, False, False)
        x_6 = None
        x_8 = x_3 + x_7
        x_3 = x_7 = None
        x_9 = torch.nn.functional.layer_norm(
            x_8,
            (432,),
            l_self_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_0_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_0_modules_norm2_parameters_bias_
        ) = None
        x_10 = torch._C._nn.linear(
            x_9,
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_9 = (
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_11 = torch._C._nn.gelu(x_10, approximate="none")
        x_10 = None
        x_12 = torch.nn.functional.dropout(x_11, 0.0, False, False)
        x_11 = None
        x_13 = torch._C._nn.linear(
            x_12,
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_12 = (
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_14 = torch.nn.functional.dropout(x_13, 0.0, False, False)
        x_13 = None
        x_15 = x_8 + x_14
        x_8 = x_14 = None
        x_16 = torch.nn.functional.layer_norm(
            x_15,
            (432,),
            l_self_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_1_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_1_modules_norm1_parameters_bias_
        ) = None
        rel_indices_1 = torch.zeros(1, 196, 196, 3)
        arange_2 = torch.arange(14)
        view_3 = arange_2.view(1, -1)
        arange_2 = None
        arange_3 = torch.arange(14)
        view_4 = arange_3.view(-1, 1)
        arange_3 = None
        ind_1 = view_3 - view_4
        view_3 = view_4 = None
        indx_1 = ind_1.repeat(14, 14)
        repeat_interleave_2 = ind_1.repeat_interleave(14, dim=0)
        ind_1 = None
        indy_1 = repeat_interleave_2.repeat_interleave(14, dim=1)
        repeat_interleave_2 = None
        pow_3 = indx_1**2
        pow_4 = indy_1**2
        indd_1 = pow_3 + pow_4
        pow_3 = pow_4 = None
        unsqueeze_4 = indd_1.unsqueeze(0)
        indd_1 = None
        rel_indices_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                2,
            )
        ] = unsqueeze_4
        setitem_3 = rel_indices_1
        unsqueeze_4 = setitem_3 = None
        unsqueeze_5 = indy_1.unsqueeze(0)
        indy_1 = None
        rel_indices_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                1,
            )
        ] = unsqueeze_5
        setitem_4 = rel_indices_1
        unsqueeze_5 = setitem_4 = None
        unsqueeze_6 = indx_1.unsqueeze(0)
        indx_1 = None
        rel_indices_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                0,
            )
        ] = unsqueeze_6
        setitem_5 = rel_indices_1
        unsqueeze_6 = setitem_5 = None
        to_1 = rel_indices_1.to(device(type="cpu"))
        rel_indices_1 = None
        linear_6 = torch._C._nn.linear(
            x_16,
            l_self_modules_blocks_modules_1_modules_attn_modules_qk_parameters_weight_,
            None,
        )
        l_self_modules_blocks_modules_1_modules_attn_modules_qk_parameters_weight_ = (
            None
        )
        reshape_3 = linear_6.reshape(1, 196, 2, 9, 48)
        linear_6 = None
        qk_1 = reshape_3.permute(2, 0, 3, 1, 4)
        reshape_3 = None
        q_1 = qk_1[0]
        k_1 = qk_1[1]
        qk_1 = None
        pos_score_3 = to_1.expand(1, -1, -1, -1)
        linear_7 = torch._C._nn.linear(
            pos_score_3,
            l_self_modules_blocks_modules_1_modules_attn_modules_pos_proj_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_attn_modules_pos_proj_parameters_bias_,
        )
        pos_score_3 = l_self_modules_blocks_modules_1_modules_attn_modules_pos_proj_parameters_weight_ = l_self_modules_blocks_modules_1_modules_attn_modules_pos_proj_parameters_bias_ = (None)
        pos_score_4 = linear_7.permute(0, 3, 1, 2)
        linear_7 = None
        transpose_3 = k_1.transpose(-2, -1)
        k_1 = None
        matmul_2 = q_1 @ transpose_3
        q_1 = transpose_3 = None
        patch_score_2 = matmul_2 * 0.14433756729740643
        matmul_2 = None
        patch_score_3 = patch_score_2.softmax(dim=-1)
        patch_score_2 = None
        pos_score_5 = pos_score_4.softmax(dim=-1)
        pos_score_4 = None
        gating_1 = (
            l_self_modules_blocks_modules_1_modules_attn_parameters_gating_param_.view(
                1, -1, 1, 1
            )
        )
        l_self_modules_blocks_modules_1_modules_attn_parameters_gating_param_ = None
        sigmoid_2 = torch.sigmoid(gating_1)
        sub_3 = 1.0 - sigmoid_2
        sigmoid_2 = None
        mul_4 = sub_3 * patch_score_3
        sub_3 = patch_score_3 = None
        sigmoid_3 = torch.sigmoid(gating_1)
        gating_1 = None
        mul_5 = sigmoid_3 * pos_score_5
        sigmoid_3 = pos_score_5 = None
        attn_3 = mul_4 + mul_5
        mul_4 = mul_5 = None
        sum_2 = attn_3.sum(dim=-1)
        unsqueeze_7 = sum_2.unsqueeze(-1)
        sum_2 = None
        attn_3 /= unsqueeze_7
        attn_4 = attn_3
        attn_3 = unsqueeze_7 = None
        attn_5 = torch.nn.functional.dropout(attn_4, 0.0, False, False)
        attn_4 = None
        linear_8 = torch._C._nn.linear(
            x_16,
            l_self_modules_blocks_modules_1_modules_attn_modules_v_parameters_weight_,
            None,
        )
        x_16 = (
            l_self_modules_blocks_modules_1_modules_attn_modules_v_parameters_weight_
        ) = None
        reshape_4 = linear_8.reshape(1, 196, 9, 48)
        linear_8 = None
        v_1 = reshape_4.permute(0, 2, 1, 3)
        reshape_4 = None
        matmul_3 = attn_5 @ v_1
        attn_5 = v_1 = None
        transpose_4 = matmul_3.transpose(1, 2)
        matmul_3 = None
        x_17 = transpose_4.reshape(1, 196, 432)
        transpose_4 = None
        x_18 = torch._C._nn.linear(
            x_17,
            l_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_17 = (
            l_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_19 = torch.nn.functional.dropout(x_18, 0.0, False, False)
        x_18 = None
        x_20 = x_15 + x_19
        x_15 = x_19 = None
        x_21 = torch.nn.functional.layer_norm(
            x_20,
            (432,),
            l_self_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_1_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_1_modules_norm2_parameters_bias_
        ) = None
        x_22 = torch._C._nn.linear(
            x_21,
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_21 = (
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_23 = torch._C._nn.gelu(x_22, approximate="none")
        x_22 = None
        x_24 = torch.nn.functional.dropout(x_23, 0.0, False, False)
        x_23 = None
        x_25 = torch._C._nn.linear(
            x_24,
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_24 = (
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_26 = torch.nn.functional.dropout(x_25, 0.0, False, False)
        x_25 = None
        x_27 = x_20 + x_26
        x_20 = x_26 = None
        x_28 = torch.nn.functional.layer_norm(
            x_27,
            (432,),
            l_self_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_2_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_2_modules_norm1_parameters_bias_
        ) = None
        rel_indices_2 = torch.zeros(1, 196, 196, 3)
        arange_4 = torch.arange(14)
        view_6 = arange_4.view(1, -1)
        arange_4 = None
        arange_5 = torch.arange(14)
        view_7 = arange_5.view(-1, 1)
        arange_5 = None
        ind_2 = view_6 - view_7
        view_6 = view_7 = None
        indx_2 = ind_2.repeat(14, 14)
        repeat_interleave_4 = ind_2.repeat_interleave(14, dim=0)
        ind_2 = None
        indy_2 = repeat_interleave_4.repeat_interleave(14, dim=1)
        repeat_interleave_4 = None
        pow_5 = indx_2**2
        pow_6 = indy_2**2
        indd_2 = pow_5 + pow_6
        pow_5 = pow_6 = None
        unsqueeze_8 = indd_2.unsqueeze(0)
        indd_2 = None
        rel_indices_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                2,
            )
        ] = unsqueeze_8
        setitem_6 = rel_indices_2
        unsqueeze_8 = setitem_6 = None
        unsqueeze_9 = indy_2.unsqueeze(0)
        indy_2 = None
        rel_indices_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                1,
            )
        ] = unsqueeze_9
        setitem_7 = rel_indices_2
        unsqueeze_9 = setitem_7 = None
        unsqueeze_10 = indx_2.unsqueeze(0)
        indx_2 = None
        rel_indices_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                0,
            )
        ] = unsqueeze_10
        setitem_8 = rel_indices_2
        unsqueeze_10 = setitem_8 = None
        to_2 = rel_indices_2.to(device(type="cpu"))
        rel_indices_2 = None
        linear_12 = torch._C._nn.linear(
            x_28,
            l_self_modules_blocks_modules_2_modules_attn_modules_qk_parameters_weight_,
            None,
        )
        l_self_modules_blocks_modules_2_modules_attn_modules_qk_parameters_weight_ = (
            None
        )
        reshape_6 = linear_12.reshape(1, 196, 2, 9, 48)
        linear_12 = None
        qk_2 = reshape_6.permute(2, 0, 3, 1, 4)
        reshape_6 = None
        q_2 = qk_2[0]
        k_2 = qk_2[1]
        qk_2 = None
        pos_score_6 = to_2.expand(1, -1, -1, -1)
        linear_13 = torch._C._nn.linear(
            pos_score_6,
            l_self_modules_blocks_modules_2_modules_attn_modules_pos_proj_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_attn_modules_pos_proj_parameters_bias_,
        )
        pos_score_6 = l_self_modules_blocks_modules_2_modules_attn_modules_pos_proj_parameters_weight_ = l_self_modules_blocks_modules_2_modules_attn_modules_pos_proj_parameters_bias_ = (None)
        pos_score_7 = linear_13.permute(0, 3, 1, 2)
        linear_13 = None
        transpose_5 = k_2.transpose(-2, -1)
        k_2 = None
        matmul_4 = q_2 @ transpose_5
        q_2 = transpose_5 = None
        patch_score_4 = matmul_4 * 0.14433756729740643
        matmul_4 = None
        patch_score_5 = patch_score_4.softmax(dim=-1)
        patch_score_4 = None
        pos_score_8 = pos_score_7.softmax(dim=-1)
        pos_score_7 = None
        gating_2 = (
            l_self_modules_blocks_modules_2_modules_attn_parameters_gating_param_.view(
                1, -1, 1, 1
            )
        )
        l_self_modules_blocks_modules_2_modules_attn_parameters_gating_param_ = None
        sigmoid_4 = torch.sigmoid(gating_2)
        sub_5 = 1.0 - sigmoid_4
        sigmoid_4 = None
        mul_7 = sub_5 * patch_score_5
        sub_5 = patch_score_5 = None
        sigmoid_5 = torch.sigmoid(gating_2)
        gating_2 = None
        mul_8 = sigmoid_5 * pos_score_8
        sigmoid_5 = pos_score_8 = None
        attn_6 = mul_7 + mul_8
        mul_7 = mul_8 = None
        sum_3 = attn_6.sum(dim=-1)
        unsqueeze_11 = sum_3.unsqueeze(-1)
        sum_3 = None
        attn_6 /= unsqueeze_11
        attn_7 = attn_6
        attn_6 = unsqueeze_11 = None
        attn_8 = torch.nn.functional.dropout(attn_7, 0.0, False, False)
        attn_7 = None
        linear_14 = torch._C._nn.linear(
            x_28,
            l_self_modules_blocks_modules_2_modules_attn_modules_v_parameters_weight_,
            None,
        )
        x_28 = (
            l_self_modules_blocks_modules_2_modules_attn_modules_v_parameters_weight_
        ) = None
        reshape_7 = linear_14.reshape(1, 196, 9, 48)
        linear_14 = None
        v_2 = reshape_7.permute(0, 2, 1, 3)
        reshape_7 = None
        matmul_5 = attn_8 @ v_2
        attn_8 = v_2 = None
        transpose_6 = matmul_5.transpose(1, 2)
        matmul_5 = None
        x_29 = transpose_6.reshape(1, 196, 432)
        transpose_6 = None
        x_30 = torch._C._nn.linear(
            x_29,
            l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_,
        )
        x_29 = (
            l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_31 = torch.nn.functional.dropout(x_30, 0.0, False, False)
        x_30 = None
        x_32 = x_27 + x_31
        x_27 = x_31 = None
        x_33 = torch.nn.functional.layer_norm(
            x_32,
            (432,),
            l_self_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_2_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_2_modules_norm2_parameters_bias_
        ) = None
        x_34 = torch._C._nn.linear(
            x_33,
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_33 = (
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_35 = torch._C._nn.gelu(x_34, approximate="none")
        x_34 = None
        x_36 = torch.nn.functional.dropout(x_35, 0.0, False, False)
        x_35 = None
        x_37 = torch._C._nn.linear(
            x_36,
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_36 = (
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_38 = torch.nn.functional.dropout(x_37, 0.0, False, False)
        x_37 = None
        x_39 = x_32 + x_38
        x_32 = x_38 = None
        x_40 = torch.nn.functional.layer_norm(
            x_39,
            (432,),
            l_self_modules_blocks_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_3_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_3_modules_norm1_parameters_bias_
        ) = None
        rel_indices_3 = torch.zeros(1, 196, 196, 3)
        arange_6 = torch.arange(14)
        view_9 = arange_6.view(1, -1)
        arange_6 = None
        arange_7 = torch.arange(14)
        view_10 = arange_7.view(-1, 1)
        arange_7 = None
        ind_3 = view_9 - view_10
        view_9 = view_10 = None
        indx_3 = ind_3.repeat(14, 14)
        repeat_interleave_6 = ind_3.repeat_interleave(14, dim=0)
        ind_3 = None
        indy_3 = repeat_interleave_6.repeat_interleave(14, dim=1)
        repeat_interleave_6 = None
        pow_7 = indx_3**2
        pow_8 = indy_3**2
        indd_3 = pow_7 + pow_8
        pow_7 = pow_8 = None
        unsqueeze_12 = indd_3.unsqueeze(0)
        indd_3 = None
        rel_indices_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                2,
            )
        ] = unsqueeze_12
        setitem_9 = rel_indices_3
        unsqueeze_12 = setitem_9 = None
        unsqueeze_13 = indy_3.unsqueeze(0)
        indy_3 = None
        rel_indices_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                1,
            )
        ] = unsqueeze_13
        setitem_10 = rel_indices_3
        unsqueeze_13 = setitem_10 = None
        unsqueeze_14 = indx_3.unsqueeze(0)
        indx_3 = None
        rel_indices_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                0,
            )
        ] = unsqueeze_14
        setitem_11 = rel_indices_3
        unsqueeze_14 = setitem_11 = None
        to_3 = rel_indices_3.to(device(type="cpu"))
        rel_indices_3 = None
        linear_18 = torch._C._nn.linear(
            x_40,
            l_self_modules_blocks_modules_3_modules_attn_modules_qk_parameters_weight_,
            None,
        )
        l_self_modules_blocks_modules_3_modules_attn_modules_qk_parameters_weight_ = (
            None
        )
        reshape_9 = linear_18.reshape(1, 196, 2, 9, 48)
        linear_18 = None
        qk_3 = reshape_9.permute(2, 0, 3, 1, 4)
        reshape_9 = None
        q_3 = qk_3[0]
        k_3 = qk_3[1]
        qk_3 = None
        pos_score_9 = to_3.expand(1, -1, -1, -1)
        linear_19 = torch._C._nn.linear(
            pos_score_9,
            l_self_modules_blocks_modules_3_modules_attn_modules_pos_proj_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_attn_modules_pos_proj_parameters_bias_,
        )
        pos_score_9 = l_self_modules_blocks_modules_3_modules_attn_modules_pos_proj_parameters_weight_ = l_self_modules_blocks_modules_3_modules_attn_modules_pos_proj_parameters_bias_ = (None)
        pos_score_10 = linear_19.permute(0, 3, 1, 2)
        linear_19 = None
        transpose_7 = k_3.transpose(-2, -1)
        k_3 = None
        matmul_6 = q_3 @ transpose_7
        q_3 = transpose_7 = None
        patch_score_6 = matmul_6 * 0.14433756729740643
        matmul_6 = None
        patch_score_7 = patch_score_6.softmax(dim=-1)
        patch_score_6 = None
        pos_score_11 = pos_score_10.softmax(dim=-1)
        pos_score_10 = None
        gating_3 = (
            l_self_modules_blocks_modules_3_modules_attn_parameters_gating_param_.view(
                1, -1, 1, 1
            )
        )
        l_self_modules_blocks_modules_3_modules_attn_parameters_gating_param_ = None
        sigmoid_6 = torch.sigmoid(gating_3)
        sub_7 = 1.0 - sigmoid_6
        sigmoid_6 = None
        mul_10 = sub_7 * patch_score_7
        sub_7 = patch_score_7 = None
        sigmoid_7 = torch.sigmoid(gating_3)
        gating_3 = None
        mul_11 = sigmoid_7 * pos_score_11
        sigmoid_7 = pos_score_11 = None
        attn_9 = mul_10 + mul_11
        mul_10 = mul_11 = None
        sum_4 = attn_9.sum(dim=-1)
        unsqueeze_15 = sum_4.unsqueeze(-1)
        sum_4 = None
        attn_9 /= unsqueeze_15
        attn_10 = attn_9
        attn_9 = unsqueeze_15 = None
        attn_11 = torch.nn.functional.dropout(attn_10, 0.0, False, False)
        attn_10 = None
        linear_20 = torch._C._nn.linear(
            x_40,
            l_self_modules_blocks_modules_3_modules_attn_modules_v_parameters_weight_,
            None,
        )
        x_40 = (
            l_self_modules_blocks_modules_3_modules_attn_modules_v_parameters_weight_
        ) = None
        reshape_10 = linear_20.reshape(1, 196, 9, 48)
        linear_20 = None
        v_3 = reshape_10.permute(0, 2, 1, 3)
        reshape_10 = None
        matmul_7 = attn_11 @ v_3
        attn_11 = v_3 = None
        transpose_8 = matmul_7.transpose(1, 2)
        matmul_7 = None
        x_41 = transpose_8.reshape(1, 196, 432)
        transpose_8 = None
        x_42 = torch._C._nn.linear(
            x_41,
            l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_,
        )
        x_41 = (
            l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_43 = torch.nn.functional.dropout(x_42, 0.0, False, False)
        x_42 = None
        x_44 = x_39 + x_43
        x_39 = x_43 = None
        x_45 = torch.nn.functional.layer_norm(
            x_44,
            (432,),
            l_self_modules_blocks_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_3_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_3_modules_norm2_parameters_bias_
        ) = None
        x_46 = torch._C._nn.linear(
            x_45,
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_45 = (
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_47 = torch._C._nn.gelu(x_46, approximate="none")
        x_46 = None
        x_48 = torch.nn.functional.dropout(x_47, 0.0, False, False)
        x_47 = None
        x_49 = torch._C._nn.linear(
            x_48,
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_48 = (
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_50 = torch.nn.functional.dropout(x_49, 0.0, False, False)
        x_49 = None
        x_51 = x_44 + x_50
        x_44 = x_50 = None
        x_52 = torch.nn.functional.layer_norm(
            x_51,
            (432,),
            l_self_modules_blocks_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_4_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_4_modules_norm1_parameters_bias_
        ) = None
        rel_indices_4 = torch.zeros(1, 196, 196, 3)
        arange_8 = torch.arange(14)
        view_12 = arange_8.view(1, -1)
        arange_8 = None
        arange_9 = torch.arange(14)
        view_13 = arange_9.view(-1, 1)
        arange_9 = None
        ind_4 = view_12 - view_13
        view_12 = view_13 = None
        indx_4 = ind_4.repeat(14, 14)
        repeat_interleave_8 = ind_4.repeat_interleave(14, dim=0)
        ind_4 = None
        indy_4 = repeat_interleave_8.repeat_interleave(14, dim=1)
        repeat_interleave_8 = None
        pow_9 = indx_4**2
        pow_10 = indy_4**2
        indd_4 = pow_9 + pow_10
        pow_9 = pow_10 = None
        unsqueeze_16 = indd_4.unsqueeze(0)
        indd_4 = None
        rel_indices_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                2,
            )
        ] = unsqueeze_16
        setitem_12 = rel_indices_4
        unsqueeze_16 = setitem_12 = None
        unsqueeze_17 = indy_4.unsqueeze(0)
        indy_4 = None
        rel_indices_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                1,
            )
        ] = unsqueeze_17
        setitem_13 = rel_indices_4
        unsqueeze_17 = setitem_13 = None
        unsqueeze_18 = indx_4.unsqueeze(0)
        indx_4 = None
        rel_indices_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                0,
            )
        ] = unsqueeze_18
        setitem_14 = rel_indices_4
        unsqueeze_18 = setitem_14 = None
        to_4 = rel_indices_4.to(device(type="cpu"))
        rel_indices_4 = None
        linear_24 = torch._C._nn.linear(
            x_52,
            l_self_modules_blocks_modules_4_modules_attn_modules_qk_parameters_weight_,
            None,
        )
        l_self_modules_blocks_modules_4_modules_attn_modules_qk_parameters_weight_ = (
            None
        )
        reshape_12 = linear_24.reshape(1, 196, 2, 9, 48)
        linear_24 = None
        qk_4 = reshape_12.permute(2, 0, 3, 1, 4)
        reshape_12 = None
        q_4 = qk_4[0]
        k_4 = qk_4[1]
        qk_4 = None
        pos_score_12 = to_4.expand(1, -1, -1, -1)
        linear_25 = torch._C._nn.linear(
            pos_score_12,
            l_self_modules_blocks_modules_4_modules_attn_modules_pos_proj_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_attn_modules_pos_proj_parameters_bias_,
        )
        pos_score_12 = l_self_modules_blocks_modules_4_modules_attn_modules_pos_proj_parameters_weight_ = l_self_modules_blocks_modules_4_modules_attn_modules_pos_proj_parameters_bias_ = (None)
        pos_score_13 = linear_25.permute(0, 3, 1, 2)
        linear_25 = None
        transpose_9 = k_4.transpose(-2, -1)
        k_4 = None
        matmul_8 = q_4 @ transpose_9
        q_4 = transpose_9 = None
        patch_score_8 = matmul_8 * 0.14433756729740643
        matmul_8 = None
        patch_score_9 = patch_score_8.softmax(dim=-1)
        patch_score_8 = None
        pos_score_14 = pos_score_13.softmax(dim=-1)
        pos_score_13 = None
        gating_4 = (
            l_self_modules_blocks_modules_4_modules_attn_parameters_gating_param_.view(
                1, -1, 1, 1
            )
        )
        l_self_modules_blocks_modules_4_modules_attn_parameters_gating_param_ = None
        sigmoid_8 = torch.sigmoid(gating_4)
        sub_9 = 1.0 - sigmoid_8
        sigmoid_8 = None
        mul_13 = sub_9 * patch_score_9
        sub_9 = patch_score_9 = None
        sigmoid_9 = torch.sigmoid(gating_4)
        gating_4 = None
        mul_14 = sigmoid_9 * pos_score_14
        sigmoid_9 = pos_score_14 = None
        attn_12 = mul_13 + mul_14
        mul_13 = mul_14 = None
        sum_5 = attn_12.sum(dim=-1)
        unsqueeze_19 = sum_5.unsqueeze(-1)
        sum_5 = None
        attn_12 /= unsqueeze_19
        attn_13 = attn_12
        attn_12 = unsqueeze_19 = None
        attn_14 = torch.nn.functional.dropout(attn_13, 0.0, False, False)
        attn_13 = None
        linear_26 = torch._C._nn.linear(
            x_52,
            l_self_modules_blocks_modules_4_modules_attn_modules_v_parameters_weight_,
            None,
        )
        x_52 = (
            l_self_modules_blocks_modules_4_modules_attn_modules_v_parameters_weight_
        ) = None
        reshape_13 = linear_26.reshape(1, 196, 9, 48)
        linear_26 = None
        v_4 = reshape_13.permute(0, 2, 1, 3)
        reshape_13 = None
        matmul_9 = attn_14 @ v_4
        attn_14 = v_4 = None
        transpose_10 = matmul_9.transpose(1, 2)
        matmul_9 = None
        x_53 = transpose_10.reshape(1, 196, 432)
        transpose_10 = None
        x_54 = torch._C._nn.linear(
            x_53,
            l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_,
        )
        x_53 = (
            l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_55 = torch.nn.functional.dropout(x_54, 0.0, False, False)
        x_54 = None
        x_56 = x_51 + x_55
        x_51 = x_55 = None
        x_57 = torch.nn.functional.layer_norm(
            x_56,
            (432,),
            l_self_modules_blocks_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_4_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_4_modules_norm2_parameters_bias_
        ) = None
        x_58 = torch._C._nn.linear(
            x_57,
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_57 = (
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_59 = torch._C._nn.gelu(x_58, approximate="none")
        x_58 = None
        x_60 = torch.nn.functional.dropout(x_59, 0.0, False, False)
        x_59 = None
        x_61 = torch._C._nn.linear(
            x_60,
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_60 = (
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_62 = torch.nn.functional.dropout(x_61, 0.0, False, False)
        x_61 = None
        x_63 = x_56 + x_62
        x_56 = x_62 = None
        x_64 = torch.nn.functional.layer_norm(
            x_63,
            (432,),
            l_self_modules_blocks_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_5_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_5_modules_norm1_parameters_bias_
        ) = None
        rel_indices_5 = torch.zeros(1, 196, 196, 3)
        arange_10 = torch.arange(14)
        view_15 = arange_10.view(1, -1)
        arange_10 = None
        arange_11 = torch.arange(14)
        view_16 = arange_11.view(-1, 1)
        arange_11 = None
        ind_5 = view_15 - view_16
        view_15 = view_16 = None
        indx_5 = ind_5.repeat(14, 14)
        repeat_interleave_10 = ind_5.repeat_interleave(14, dim=0)
        ind_5 = None
        indy_5 = repeat_interleave_10.repeat_interleave(14, dim=1)
        repeat_interleave_10 = None
        pow_11 = indx_5**2
        pow_12 = indy_5**2
        indd_5 = pow_11 + pow_12
        pow_11 = pow_12 = None
        unsqueeze_20 = indd_5.unsqueeze(0)
        indd_5 = None
        rel_indices_5[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                2,
            )
        ] = unsqueeze_20
        setitem_15 = rel_indices_5
        unsqueeze_20 = setitem_15 = None
        unsqueeze_21 = indy_5.unsqueeze(0)
        indy_5 = None
        rel_indices_5[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                1,
            )
        ] = unsqueeze_21
        setitem_16 = rel_indices_5
        unsqueeze_21 = setitem_16 = None
        unsqueeze_22 = indx_5.unsqueeze(0)
        indx_5 = None
        rel_indices_5[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                0,
            )
        ] = unsqueeze_22
        setitem_17 = rel_indices_5
        unsqueeze_22 = setitem_17 = None
        to_5 = rel_indices_5.to(device(type="cpu"))
        rel_indices_5 = None
        linear_30 = torch._C._nn.linear(
            x_64,
            l_self_modules_blocks_modules_5_modules_attn_modules_qk_parameters_weight_,
            None,
        )
        l_self_modules_blocks_modules_5_modules_attn_modules_qk_parameters_weight_ = (
            None
        )
        reshape_15 = linear_30.reshape(1, 196, 2, 9, 48)
        linear_30 = None
        qk_5 = reshape_15.permute(2, 0, 3, 1, 4)
        reshape_15 = None
        q_5 = qk_5[0]
        k_5 = qk_5[1]
        qk_5 = None
        pos_score_15 = to_5.expand(1, -1, -1, -1)
        linear_31 = torch._C._nn.linear(
            pos_score_15,
            l_self_modules_blocks_modules_5_modules_attn_modules_pos_proj_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_attn_modules_pos_proj_parameters_bias_,
        )
        pos_score_15 = l_self_modules_blocks_modules_5_modules_attn_modules_pos_proj_parameters_weight_ = l_self_modules_blocks_modules_5_modules_attn_modules_pos_proj_parameters_bias_ = (None)
        pos_score_16 = linear_31.permute(0, 3, 1, 2)
        linear_31 = None
        transpose_11 = k_5.transpose(-2, -1)
        k_5 = None
        matmul_10 = q_5 @ transpose_11
        q_5 = transpose_11 = None
        patch_score_10 = matmul_10 * 0.14433756729740643
        matmul_10 = None
        patch_score_11 = patch_score_10.softmax(dim=-1)
        patch_score_10 = None
        pos_score_17 = pos_score_16.softmax(dim=-1)
        pos_score_16 = None
        gating_5 = (
            l_self_modules_blocks_modules_5_modules_attn_parameters_gating_param_.view(
                1, -1, 1, 1
            )
        )
        l_self_modules_blocks_modules_5_modules_attn_parameters_gating_param_ = None
        sigmoid_10 = torch.sigmoid(gating_5)
        sub_11 = 1.0 - sigmoid_10
        sigmoid_10 = None
        mul_16 = sub_11 * patch_score_11
        sub_11 = patch_score_11 = None
        sigmoid_11 = torch.sigmoid(gating_5)
        gating_5 = None
        mul_17 = sigmoid_11 * pos_score_17
        sigmoid_11 = pos_score_17 = None
        attn_15 = mul_16 + mul_17
        mul_16 = mul_17 = None
        sum_6 = attn_15.sum(dim=-1)
        unsqueeze_23 = sum_6.unsqueeze(-1)
        sum_6 = None
        attn_15 /= unsqueeze_23
        attn_16 = attn_15
        attn_15 = unsqueeze_23 = None
        attn_17 = torch.nn.functional.dropout(attn_16, 0.0, False, False)
        attn_16 = None
        linear_32 = torch._C._nn.linear(
            x_64,
            l_self_modules_blocks_modules_5_modules_attn_modules_v_parameters_weight_,
            None,
        )
        x_64 = (
            l_self_modules_blocks_modules_5_modules_attn_modules_v_parameters_weight_
        ) = None
        reshape_16 = linear_32.reshape(1, 196, 9, 48)
        linear_32 = None
        v_5 = reshape_16.permute(0, 2, 1, 3)
        reshape_16 = None
        matmul_11 = attn_17 @ v_5
        attn_17 = v_5 = None
        transpose_12 = matmul_11.transpose(1, 2)
        matmul_11 = None
        x_65 = transpose_12.reshape(1, 196, 432)
        transpose_12 = None
        x_66 = torch._C._nn.linear(
            x_65,
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_,
        )
        x_65 = (
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_67 = torch.nn.functional.dropout(x_66, 0.0, False, False)
        x_66 = None
        x_68 = x_63 + x_67
        x_63 = x_67 = None
        x_69 = torch.nn.functional.layer_norm(
            x_68,
            (432,),
            l_self_modules_blocks_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_5_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_5_modules_norm2_parameters_bias_
        ) = None
        x_70 = torch._C._nn.linear(
            x_69,
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_69 = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_71 = torch._C._nn.gelu(x_70, approximate="none")
        x_70 = None
        x_72 = torch.nn.functional.dropout(x_71, 0.0, False, False)
        x_71 = None
        x_73 = torch._C._nn.linear(
            x_72,
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_72 = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_74 = torch.nn.functional.dropout(x_73, 0.0, False, False)
        x_73 = None
        x_75 = x_68 + x_74
        x_68 = x_74 = None
        x_76 = torch.nn.functional.layer_norm(
            x_75,
            (432,),
            l_self_modules_blocks_modules_6_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_6_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_6_modules_norm1_parameters_bias_
        ) = None
        rel_indices_6 = torch.zeros(1, 196, 196, 3)
        arange_12 = torch.arange(14)
        view_18 = arange_12.view(1, -1)
        arange_12 = None
        arange_13 = torch.arange(14)
        view_19 = arange_13.view(-1, 1)
        arange_13 = None
        ind_6 = view_18 - view_19
        view_18 = view_19 = None
        indx_6 = ind_6.repeat(14, 14)
        repeat_interleave_12 = ind_6.repeat_interleave(14, dim=0)
        ind_6 = None
        indy_6 = repeat_interleave_12.repeat_interleave(14, dim=1)
        repeat_interleave_12 = None
        pow_13 = indx_6**2
        pow_14 = indy_6**2
        indd_6 = pow_13 + pow_14
        pow_13 = pow_14 = None
        unsqueeze_24 = indd_6.unsqueeze(0)
        indd_6 = None
        rel_indices_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                2,
            )
        ] = unsqueeze_24
        setitem_18 = rel_indices_6
        unsqueeze_24 = setitem_18 = None
        unsqueeze_25 = indy_6.unsqueeze(0)
        indy_6 = None
        rel_indices_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                1,
            )
        ] = unsqueeze_25
        setitem_19 = rel_indices_6
        unsqueeze_25 = setitem_19 = None
        unsqueeze_26 = indx_6.unsqueeze(0)
        indx_6 = None
        rel_indices_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                0,
            )
        ] = unsqueeze_26
        setitem_20 = rel_indices_6
        unsqueeze_26 = setitem_20 = None
        to_6 = rel_indices_6.to(device(type="cpu"))
        rel_indices_6 = None
        linear_36 = torch._C._nn.linear(
            x_76,
            l_self_modules_blocks_modules_6_modules_attn_modules_qk_parameters_weight_,
            None,
        )
        l_self_modules_blocks_modules_6_modules_attn_modules_qk_parameters_weight_ = (
            None
        )
        reshape_18 = linear_36.reshape(1, 196, 2, 9, 48)
        linear_36 = None
        qk_6 = reshape_18.permute(2, 0, 3, 1, 4)
        reshape_18 = None
        q_6 = qk_6[0]
        k_6 = qk_6[1]
        qk_6 = None
        pos_score_18 = to_6.expand(1, -1, -1, -1)
        linear_37 = torch._C._nn.linear(
            pos_score_18,
            l_self_modules_blocks_modules_6_modules_attn_modules_pos_proj_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_attn_modules_pos_proj_parameters_bias_,
        )
        pos_score_18 = l_self_modules_blocks_modules_6_modules_attn_modules_pos_proj_parameters_weight_ = l_self_modules_blocks_modules_6_modules_attn_modules_pos_proj_parameters_bias_ = (None)
        pos_score_19 = linear_37.permute(0, 3, 1, 2)
        linear_37 = None
        transpose_13 = k_6.transpose(-2, -1)
        k_6 = None
        matmul_12 = q_6 @ transpose_13
        q_6 = transpose_13 = None
        patch_score_12 = matmul_12 * 0.14433756729740643
        matmul_12 = None
        patch_score_13 = patch_score_12.softmax(dim=-1)
        patch_score_12 = None
        pos_score_20 = pos_score_19.softmax(dim=-1)
        pos_score_19 = None
        gating_6 = (
            l_self_modules_blocks_modules_6_modules_attn_parameters_gating_param_.view(
                1, -1, 1, 1
            )
        )
        l_self_modules_blocks_modules_6_modules_attn_parameters_gating_param_ = None
        sigmoid_12 = torch.sigmoid(gating_6)
        sub_13 = 1.0 - sigmoid_12
        sigmoid_12 = None
        mul_19 = sub_13 * patch_score_13
        sub_13 = patch_score_13 = None
        sigmoid_13 = torch.sigmoid(gating_6)
        gating_6 = None
        mul_20 = sigmoid_13 * pos_score_20
        sigmoid_13 = pos_score_20 = None
        attn_18 = mul_19 + mul_20
        mul_19 = mul_20 = None
        sum_7 = attn_18.sum(dim=-1)
        unsqueeze_27 = sum_7.unsqueeze(-1)
        sum_7 = None
        attn_18 /= unsqueeze_27
        attn_19 = attn_18
        attn_18 = unsqueeze_27 = None
        attn_20 = torch.nn.functional.dropout(attn_19, 0.0, False, False)
        attn_19 = None
        linear_38 = torch._C._nn.linear(
            x_76,
            l_self_modules_blocks_modules_6_modules_attn_modules_v_parameters_weight_,
            None,
        )
        x_76 = (
            l_self_modules_blocks_modules_6_modules_attn_modules_v_parameters_weight_
        ) = None
        reshape_19 = linear_38.reshape(1, 196, 9, 48)
        linear_38 = None
        v_6 = reshape_19.permute(0, 2, 1, 3)
        reshape_19 = None
        matmul_13 = attn_20 @ v_6
        attn_20 = v_6 = None
        transpose_14 = matmul_13.transpose(1, 2)
        matmul_13 = None
        x_77 = transpose_14.reshape(1, 196, 432)
        transpose_14 = None
        x_78 = torch._C._nn.linear(
            x_77,
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_,
        )
        x_77 = (
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_79 = torch.nn.functional.dropout(x_78, 0.0, False, False)
        x_78 = None
        x_80 = x_75 + x_79
        x_75 = x_79 = None
        x_81 = torch.nn.functional.layer_norm(
            x_80,
            (432,),
            l_self_modules_blocks_modules_6_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_6_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_6_modules_norm2_parameters_bias_
        ) = None
        x_82 = torch._C._nn.linear(
            x_81,
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_81 = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_83 = torch._C._nn.gelu(x_82, approximate="none")
        x_82 = None
        x_84 = torch.nn.functional.dropout(x_83, 0.0, False, False)
        x_83 = None
        x_85 = torch._C._nn.linear(
            x_84,
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_84 = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_86 = torch.nn.functional.dropout(x_85, 0.0, False, False)
        x_85 = None
        x_87 = x_80 + x_86
        x_80 = x_86 = None
        x_88 = torch.nn.functional.layer_norm(
            x_87,
            (432,),
            l_self_modules_blocks_modules_7_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_7_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_7_modules_norm1_parameters_bias_
        ) = None
        rel_indices_7 = torch.zeros(1, 196, 196, 3)
        arange_14 = torch.arange(14)
        view_21 = arange_14.view(1, -1)
        arange_14 = None
        arange_15 = torch.arange(14)
        view_22 = arange_15.view(-1, 1)
        arange_15 = None
        ind_7 = view_21 - view_22
        view_21 = view_22 = None
        indx_7 = ind_7.repeat(14, 14)
        repeat_interleave_14 = ind_7.repeat_interleave(14, dim=0)
        ind_7 = None
        indy_7 = repeat_interleave_14.repeat_interleave(14, dim=1)
        repeat_interleave_14 = None
        pow_15 = indx_7**2
        pow_16 = indy_7**2
        indd_7 = pow_15 + pow_16
        pow_15 = pow_16 = None
        unsqueeze_28 = indd_7.unsqueeze(0)
        indd_7 = None
        rel_indices_7[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                2,
            )
        ] = unsqueeze_28
        setitem_21 = rel_indices_7
        unsqueeze_28 = setitem_21 = None
        unsqueeze_29 = indy_7.unsqueeze(0)
        indy_7 = None
        rel_indices_7[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                1,
            )
        ] = unsqueeze_29
        setitem_22 = rel_indices_7
        unsqueeze_29 = setitem_22 = None
        unsqueeze_30 = indx_7.unsqueeze(0)
        indx_7 = None
        rel_indices_7[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                0,
            )
        ] = unsqueeze_30
        setitem_23 = rel_indices_7
        unsqueeze_30 = setitem_23 = None
        to_7 = rel_indices_7.to(device(type="cpu"))
        rel_indices_7 = None
        linear_42 = torch._C._nn.linear(
            x_88,
            l_self_modules_blocks_modules_7_modules_attn_modules_qk_parameters_weight_,
            None,
        )
        l_self_modules_blocks_modules_7_modules_attn_modules_qk_parameters_weight_ = (
            None
        )
        reshape_21 = linear_42.reshape(1, 196, 2, 9, 48)
        linear_42 = None
        qk_7 = reshape_21.permute(2, 0, 3, 1, 4)
        reshape_21 = None
        q_7 = qk_7[0]
        k_7 = qk_7[1]
        qk_7 = None
        pos_score_21 = to_7.expand(1, -1, -1, -1)
        linear_43 = torch._C._nn.linear(
            pos_score_21,
            l_self_modules_blocks_modules_7_modules_attn_modules_pos_proj_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_attn_modules_pos_proj_parameters_bias_,
        )
        pos_score_21 = l_self_modules_blocks_modules_7_modules_attn_modules_pos_proj_parameters_weight_ = l_self_modules_blocks_modules_7_modules_attn_modules_pos_proj_parameters_bias_ = (None)
        pos_score_22 = linear_43.permute(0, 3, 1, 2)
        linear_43 = None
        transpose_15 = k_7.transpose(-2, -1)
        k_7 = None
        matmul_14 = q_7 @ transpose_15
        q_7 = transpose_15 = None
        patch_score_14 = matmul_14 * 0.14433756729740643
        matmul_14 = None
        patch_score_15 = patch_score_14.softmax(dim=-1)
        patch_score_14 = None
        pos_score_23 = pos_score_22.softmax(dim=-1)
        pos_score_22 = None
        gating_7 = (
            l_self_modules_blocks_modules_7_modules_attn_parameters_gating_param_.view(
                1, -1, 1, 1
            )
        )
        l_self_modules_blocks_modules_7_modules_attn_parameters_gating_param_ = None
        sigmoid_14 = torch.sigmoid(gating_7)
        sub_15 = 1.0 - sigmoid_14
        sigmoid_14 = None
        mul_22 = sub_15 * patch_score_15
        sub_15 = patch_score_15 = None
        sigmoid_15 = torch.sigmoid(gating_7)
        gating_7 = None
        mul_23 = sigmoid_15 * pos_score_23
        sigmoid_15 = pos_score_23 = None
        attn_21 = mul_22 + mul_23
        mul_22 = mul_23 = None
        sum_8 = attn_21.sum(dim=-1)
        unsqueeze_31 = sum_8.unsqueeze(-1)
        sum_8 = None
        attn_21 /= unsqueeze_31
        attn_22 = attn_21
        attn_21 = unsqueeze_31 = None
        attn_23 = torch.nn.functional.dropout(attn_22, 0.0, False, False)
        attn_22 = None
        linear_44 = torch._C._nn.linear(
            x_88,
            l_self_modules_blocks_modules_7_modules_attn_modules_v_parameters_weight_,
            None,
        )
        x_88 = (
            l_self_modules_blocks_modules_7_modules_attn_modules_v_parameters_weight_
        ) = None
        reshape_22 = linear_44.reshape(1, 196, 9, 48)
        linear_44 = None
        v_7 = reshape_22.permute(0, 2, 1, 3)
        reshape_22 = None
        matmul_15 = attn_23 @ v_7
        attn_23 = v_7 = None
        transpose_16 = matmul_15.transpose(1, 2)
        matmul_15 = None
        x_89 = transpose_16.reshape(1, 196, 432)
        transpose_16 = None
        x_90 = torch._C._nn.linear(
            x_89,
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_,
        )
        x_89 = (
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_91 = torch.nn.functional.dropout(x_90, 0.0, False, False)
        x_90 = None
        x_92 = x_87 + x_91
        x_87 = x_91 = None
        x_93 = torch.nn.functional.layer_norm(
            x_92,
            (432,),
            l_self_modules_blocks_modules_7_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_7_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_7_modules_norm2_parameters_bias_
        ) = None
        x_94 = torch._C._nn.linear(
            x_93,
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_93 = (
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_95 = torch._C._nn.gelu(x_94, approximate="none")
        x_94 = None
        x_96 = torch.nn.functional.dropout(x_95, 0.0, False, False)
        x_95 = None
        x_97 = torch._C._nn.linear(
            x_96,
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_96 = (
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_98 = torch.nn.functional.dropout(x_97, 0.0, False, False)
        x_97 = None
        x_99 = x_92 + x_98
        x_92 = x_98 = None
        x_100 = torch.nn.functional.layer_norm(
            x_99,
            (432,),
            l_self_modules_blocks_modules_8_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_8_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_8_modules_norm1_parameters_bias_
        ) = None
        rel_indices_8 = torch.zeros(1, 196, 196, 3)
        arange_16 = torch.arange(14)
        view_24 = arange_16.view(1, -1)
        arange_16 = None
        arange_17 = torch.arange(14)
        view_25 = arange_17.view(-1, 1)
        arange_17 = None
        ind_8 = view_24 - view_25
        view_24 = view_25 = None
        indx_8 = ind_8.repeat(14, 14)
        repeat_interleave_16 = ind_8.repeat_interleave(14, dim=0)
        ind_8 = None
        indy_8 = repeat_interleave_16.repeat_interleave(14, dim=1)
        repeat_interleave_16 = None
        pow_17 = indx_8**2
        pow_18 = indy_8**2
        indd_8 = pow_17 + pow_18
        pow_17 = pow_18 = None
        unsqueeze_32 = indd_8.unsqueeze(0)
        indd_8 = None
        rel_indices_8[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                2,
            )
        ] = unsqueeze_32
        setitem_24 = rel_indices_8
        unsqueeze_32 = setitem_24 = None
        unsqueeze_33 = indy_8.unsqueeze(0)
        indy_8 = None
        rel_indices_8[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                1,
            )
        ] = unsqueeze_33
        setitem_25 = rel_indices_8
        unsqueeze_33 = setitem_25 = None
        unsqueeze_34 = indx_8.unsqueeze(0)
        indx_8 = None
        rel_indices_8[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                0,
            )
        ] = unsqueeze_34
        setitem_26 = rel_indices_8
        unsqueeze_34 = setitem_26 = None
        to_8 = rel_indices_8.to(device(type="cpu"))
        rel_indices_8 = None
        linear_48 = torch._C._nn.linear(
            x_100,
            l_self_modules_blocks_modules_8_modules_attn_modules_qk_parameters_weight_,
            None,
        )
        l_self_modules_blocks_modules_8_modules_attn_modules_qk_parameters_weight_ = (
            None
        )
        reshape_24 = linear_48.reshape(1, 196, 2, 9, 48)
        linear_48 = None
        qk_8 = reshape_24.permute(2, 0, 3, 1, 4)
        reshape_24 = None
        q_8 = qk_8[0]
        k_8 = qk_8[1]
        qk_8 = None
        pos_score_24 = to_8.expand(1, -1, -1, -1)
        linear_49 = torch._C._nn.linear(
            pos_score_24,
            l_self_modules_blocks_modules_8_modules_attn_modules_pos_proj_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_attn_modules_pos_proj_parameters_bias_,
        )
        pos_score_24 = l_self_modules_blocks_modules_8_modules_attn_modules_pos_proj_parameters_weight_ = l_self_modules_blocks_modules_8_modules_attn_modules_pos_proj_parameters_bias_ = (None)
        pos_score_25 = linear_49.permute(0, 3, 1, 2)
        linear_49 = None
        transpose_17 = k_8.transpose(-2, -1)
        k_8 = None
        matmul_16 = q_8 @ transpose_17
        q_8 = transpose_17 = None
        patch_score_16 = matmul_16 * 0.14433756729740643
        matmul_16 = None
        patch_score_17 = patch_score_16.softmax(dim=-1)
        patch_score_16 = None
        pos_score_26 = pos_score_25.softmax(dim=-1)
        pos_score_25 = None
        gating_8 = (
            l_self_modules_blocks_modules_8_modules_attn_parameters_gating_param_.view(
                1, -1, 1, 1
            )
        )
        l_self_modules_blocks_modules_8_modules_attn_parameters_gating_param_ = None
        sigmoid_16 = torch.sigmoid(gating_8)
        sub_17 = 1.0 - sigmoid_16
        sigmoid_16 = None
        mul_25 = sub_17 * patch_score_17
        sub_17 = patch_score_17 = None
        sigmoid_17 = torch.sigmoid(gating_8)
        gating_8 = None
        mul_26 = sigmoid_17 * pos_score_26
        sigmoid_17 = pos_score_26 = None
        attn_24 = mul_25 + mul_26
        mul_25 = mul_26 = None
        sum_9 = attn_24.sum(dim=-1)
        unsqueeze_35 = sum_9.unsqueeze(-1)
        sum_9 = None
        attn_24 /= unsqueeze_35
        attn_25 = attn_24
        attn_24 = unsqueeze_35 = None
        attn_26 = torch.nn.functional.dropout(attn_25, 0.0, False, False)
        attn_25 = None
        linear_50 = torch._C._nn.linear(
            x_100,
            l_self_modules_blocks_modules_8_modules_attn_modules_v_parameters_weight_,
            None,
        )
        x_100 = (
            l_self_modules_blocks_modules_8_modules_attn_modules_v_parameters_weight_
        ) = None
        reshape_25 = linear_50.reshape(1, 196, 9, 48)
        linear_50 = None
        v_8 = reshape_25.permute(0, 2, 1, 3)
        reshape_25 = None
        matmul_17 = attn_26 @ v_8
        attn_26 = v_8 = None
        transpose_18 = matmul_17.transpose(1, 2)
        matmul_17 = None
        x_101 = transpose_18.reshape(1, 196, 432)
        transpose_18 = None
        x_102 = torch._C._nn.linear(
            x_101,
            l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_,
        )
        x_101 = (
            l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_103 = torch.nn.functional.dropout(x_102, 0.0, False, False)
        x_102 = None
        x_104 = x_99 + x_103
        x_99 = x_103 = None
        x_105 = torch.nn.functional.layer_norm(
            x_104,
            (432,),
            l_self_modules_blocks_modules_8_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_8_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_8_modules_norm2_parameters_bias_
        ) = None
        x_106 = torch._C._nn.linear(
            x_105,
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_105 = (
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_107 = torch._C._nn.gelu(x_106, approximate="none")
        x_106 = None
        x_108 = torch.nn.functional.dropout(x_107, 0.0, False, False)
        x_107 = None
        x_109 = torch._C._nn.linear(
            x_108,
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_108 = (
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_110 = torch.nn.functional.dropout(x_109, 0.0, False, False)
        x_109 = None
        x_111 = x_104 + x_110
        x_104 = x_110 = None
        x_112 = torch.nn.functional.layer_norm(
            x_111,
            (432,),
            l_self_modules_blocks_modules_9_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_9_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_9_modules_norm1_parameters_bias_
        ) = None
        rel_indices_9 = torch.zeros(1, 196, 196, 3)
        arange_18 = torch.arange(14)
        view_27 = arange_18.view(1, -1)
        arange_18 = None
        arange_19 = torch.arange(14)
        view_28 = arange_19.view(-1, 1)
        arange_19 = None
        ind_9 = view_27 - view_28
        view_27 = view_28 = None
        indx_9 = ind_9.repeat(14, 14)
        repeat_interleave_18 = ind_9.repeat_interleave(14, dim=0)
        ind_9 = None
        indy_9 = repeat_interleave_18.repeat_interleave(14, dim=1)
        repeat_interleave_18 = None
        pow_19 = indx_9**2
        pow_20 = indy_9**2
        indd_9 = pow_19 + pow_20
        pow_19 = pow_20 = None
        unsqueeze_36 = indd_9.unsqueeze(0)
        indd_9 = None
        rel_indices_9[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                2,
            )
        ] = unsqueeze_36
        setitem_27 = rel_indices_9
        unsqueeze_36 = setitem_27 = None
        unsqueeze_37 = indy_9.unsqueeze(0)
        indy_9 = None
        rel_indices_9[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                1,
            )
        ] = unsqueeze_37
        setitem_28 = rel_indices_9
        unsqueeze_37 = setitem_28 = None
        unsqueeze_38 = indx_9.unsqueeze(0)
        indx_9 = None
        rel_indices_9[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                0,
            )
        ] = unsqueeze_38
        setitem_29 = rel_indices_9
        unsqueeze_38 = setitem_29 = None
        to_9 = rel_indices_9.to(device(type="cpu"))
        rel_indices_9 = None
        linear_54 = torch._C._nn.linear(
            x_112,
            l_self_modules_blocks_modules_9_modules_attn_modules_qk_parameters_weight_,
            None,
        )
        l_self_modules_blocks_modules_9_modules_attn_modules_qk_parameters_weight_ = (
            None
        )
        reshape_27 = linear_54.reshape(1, 196, 2, 9, 48)
        linear_54 = None
        qk_9 = reshape_27.permute(2, 0, 3, 1, 4)
        reshape_27 = None
        q_9 = qk_9[0]
        k_9 = qk_9[1]
        qk_9 = None
        pos_score_27 = to_9.expand(1, -1, -1, -1)
        linear_55 = torch._C._nn.linear(
            pos_score_27,
            l_self_modules_blocks_modules_9_modules_attn_modules_pos_proj_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_attn_modules_pos_proj_parameters_bias_,
        )
        pos_score_27 = l_self_modules_blocks_modules_9_modules_attn_modules_pos_proj_parameters_weight_ = l_self_modules_blocks_modules_9_modules_attn_modules_pos_proj_parameters_bias_ = (None)
        pos_score_28 = linear_55.permute(0, 3, 1, 2)
        linear_55 = None
        transpose_19 = k_9.transpose(-2, -1)
        k_9 = None
        matmul_18 = q_9 @ transpose_19
        q_9 = transpose_19 = None
        patch_score_18 = matmul_18 * 0.14433756729740643
        matmul_18 = None
        patch_score_19 = patch_score_18.softmax(dim=-1)
        patch_score_18 = None
        pos_score_29 = pos_score_28.softmax(dim=-1)
        pos_score_28 = None
        gating_9 = (
            l_self_modules_blocks_modules_9_modules_attn_parameters_gating_param_.view(
                1, -1, 1, 1
            )
        )
        l_self_modules_blocks_modules_9_modules_attn_parameters_gating_param_ = None
        sigmoid_18 = torch.sigmoid(gating_9)
        sub_19 = 1.0 - sigmoid_18
        sigmoid_18 = None
        mul_28 = sub_19 * patch_score_19
        sub_19 = patch_score_19 = None
        sigmoid_19 = torch.sigmoid(gating_9)
        gating_9 = None
        mul_29 = sigmoid_19 * pos_score_29
        sigmoid_19 = pos_score_29 = None
        attn_27 = mul_28 + mul_29
        mul_28 = mul_29 = None
        sum_10 = attn_27.sum(dim=-1)
        unsqueeze_39 = sum_10.unsqueeze(-1)
        sum_10 = None
        attn_27 /= unsqueeze_39
        attn_28 = attn_27
        attn_27 = unsqueeze_39 = None
        attn_29 = torch.nn.functional.dropout(attn_28, 0.0, False, False)
        attn_28 = None
        linear_56 = torch._C._nn.linear(
            x_112,
            l_self_modules_blocks_modules_9_modules_attn_modules_v_parameters_weight_,
            None,
        )
        x_112 = (
            l_self_modules_blocks_modules_9_modules_attn_modules_v_parameters_weight_
        ) = None
        reshape_28 = linear_56.reshape(1, 196, 9, 48)
        linear_56 = None
        v_9 = reshape_28.permute(0, 2, 1, 3)
        reshape_28 = None
        matmul_19 = attn_29 @ v_9
        attn_29 = v_9 = None
        transpose_20 = matmul_19.transpose(1, 2)
        matmul_19 = None
        x_113 = transpose_20.reshape(1, 196, 432)
        transpose_20 = None
        x_114 = torch._C._nn.linear(
            x_113,
            l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_,
        )
        x_113 = (
            l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_115 = torch.nn.functional.dropout(x_114, 0.0, False, False)
        x_114 = None
        x_116 = x_111 + x_115
        x_111 = x_115 = None
        x_117 = torch.nn.functional.layer_norm(
            x_116,
            (432,),
            l_self_modules_blocks_modules_9_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_9_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_9_modules_norm2_parameters_bias_
        ) = None
        x_118 = torch._C._nn.linear(
            x_117,
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_117 = (
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_119 = torch._C._nn.gelu(x_118, approximate="none")
        x_118 = None
        x_120 = torch.nn.functional.dropout(x_119, 0.0, False, False)
        x_119 = None
        x_121 = torch._C._nn.linear(
            x_120,
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_120 = (
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_122 = torch.nn.functional.dropout(x_121, 0.0, False, False)
        x_121 = None
        x_123 = x_116 + x_122
        x_116 = x_122 = None
        x_124 = torch.cat((cls_tokens, x_123), dim=1)
        cls_tokens = x_123 = None
        x_125 = torch.nn.functional.layer_norm(
            x_124,
            (432,),
            l_self_modules_blocks_modules_10_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_10_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_10_modules_norm1_parameters_bias_
        ) = None
        linear_60 = torch._C._nn.linear(
            x_125,
            l_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_,
            None,
        )
        x_125 = (
            l_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_
        ) = None
        reshape_30 = linear_60.reshape(1, 197, 3, 9, 48)
        linear_60 = None
        qkv = reshape_30.permute(2, 0, 3, 1, 4)
        reshape_30 = None
        unbind = qkv.unbind(0)
        qkv = None
        q_10 = unbind[0]
        k_10 = unbind[1]
        v_10 = unbind[2]
        unbind = None
        transpose_21 = k_10.transpose(-2, -1)
        k_10 = None
        matmul_20 = q_10 @ transpose_21
        q_10 = transpose_21 = None
        attn_30 = matmul_20 * 0.14433756729740643
        matmul_20 = None
        attn_31 = attn_30.softmax(dim=-1)
        attn_30 = None
        attn_32 = torch.nn.functional.dropout(attn_31, 0.0, False, False)
        attn_31 = None
        matmul_21 = attn_32 @ v_10
        attn_32 = v_10 = None
        transpose_22 = matmul_21.transpose(1, 2)
        matmul_21 = None
        x_126 = transpose_22.reshape(1, 197, 432)
        transpose_22 = None
        x_127 = torch._C._nn.linear(
            x_126,
            l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_,
        )
        x_126 = l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_128 = torch.nn.functional.dropout(x_127, 0.0, False, False)
        x_127 = None
        x_129 = x_124 + x_128
        x_124 = x_128 = None
        x_130 = torch.nn.functional.layer_norm(
            x_129,
            (432,),
            l_self_modules_blocks_modules_10_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_10_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_10_modules_norm2_parameters_bias_
        ) = None
        x_131 = torch._C._nn.linear(
            x_130,
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_130 = (
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_132 = torch._C._nn.gelu(x_131, approximate="none")
        x_131 = None
        x_133 = torch.nn.functional.dropout(x_132, 0.0, False, False)
        x_132 = None
        x_134 = torch._C._nn.linear(
            x_133,
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_133 = (
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_135 = torch.nn.functional.dropout(x_134, 0.0, False, False)
        x_134 = None
        x_136 = x_129 + x_135
        x_129 = x_135 = None
        x_137 = torch.nn.functional.layer_norm(
            x_136,
            (432,),
            l_self_modules_blocks_modules_11_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_11_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_11_modules_norm1_parameters_bias_
        ) = None
        linear_64 = torch._C._nn.linear(
            x_137,
            l_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_,
            None,
        )
        x_137 = (
            l_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_
        ) = None
        reshape_32 = linear_64.reshape(1, 197, 3, 9, 48)
        linear_64 = None
        qkv_1 = reshape_32.permute(2, 0, 3, 1, 4)
        reshape_32 = None
        unbind_1 = qkv_1.unbind(0)
        qkv_1 = None
        q_11 = unbind_1[0]
        k_11 = unbind_1[1]
        v_11 = unbind_1[2]
        unbind_1 = None
        transpose_23 = k_11.transpose(-2, -1)
        k_11 = None
        matmul_22 = q_11 @ transpose_23
        q_11 = transpose_23 = None
        attn_33 = matmul_22 * 0.14433756729740643
        matmul_22 = None
        attn_34 = attn_33.softmax(dim=-1)
        attn_33 = None
        attn_35 = torch.nn.functional.dropout(attn_34, 0.0, False, False)
        attn_34 = None
        matmul_23 = attn_35 @ v_11
        attn_35 = v_11 = None
        transpose_24 = matmul_23.transpose(1, 2)
        matmul_23 = None
        x_138 = transpose_24.reshape(1, 197, 432)
        transpose_24 = None
        x_139 = torch._C._nn.linear(
            x_138,
            l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_,
        )
        x_138 = l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_140 = torch.nn.functional.dropout(x_139, 0.0, False, False)
        x_139 = None
        x_141 = x_136 + x_140
        x_136 = x_140 = None
        x_142 = torch.nn.functional.layer_norm(
            x_141,
            (432,),
            l_self_modules_blocks_modules_11_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_11_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_11_modules_norm2_parameters_bias_
        ) = None
        x_143 = torch._C._nn.linear(
            x_142,
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_,
        )
        x_142 = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_144 = torch._C._nn.gelu(x_143, approximate="none")
        x_143 = None
        x_145 = torch.nn.functional.dropout(x_144, 0.0, False, False)
        x_144 = None
        x_146 = torch._C._nn.linear(
            x_145,
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_145 = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_147 = torch.nn.functional.dropout(x_146, 0.0, False, False)
        x_146 = None
        x_148 = x_141 + x_147
        x_141 = x_147 = None
        x_149 = torch.nn.functional.layer_norm(
            x_148,
            (432,),
            l_self_modules_norm_parameters_weight_,
            l_self_modules_norm_parameters_bias_,
            1e-06,
        )
        x_148 = (
            l_self_modules_norm_parameters_weight_
        ) = l_self_modules_norm_parameters_bias_ = None
        x_150 = x_149[(slice(None, None, None), 0)]
        x_149 = None
        x_151 = torch.nn.functional.dropout(x_150, 0.0, False, False)
        x_150 = None
        x_152 = torch._C._nn.linear(
            x_151,
            l_self_modules_head_parameters_weight_,
            l_self_modules_head_parameters_bias_,
        )
        x_151 = (
            l_self_modules_head_parameters_weight_
        ) = l_self_modules_head_parameters_bias_ = None
        return (x_152, to, to_1, to_2, to_3, to_4, to_5, to_6, to_7, to_8, to_9)
